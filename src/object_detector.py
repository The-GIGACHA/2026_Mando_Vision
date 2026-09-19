#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
object_detector.py — 콘 / 배달표지판 공용 YOLO 검출 노드 (2026)

원본: 2025-CREAM-ERP42 의 cone_detector.py + traffic_sign_detector.py
      두 파일이 거의 같은 일을 하고 있어서 하나로 합쳤습니다.

2025 대비 바뀐 점
  1. ★ os.system("pip install ultralytics") 삭제.
     2025 cone_detector.py L20 에 있었습니다. 대회장에 네트워크가 없으면
     여기서 멈추고, 규정 12번(1분 이상 정지)에 걸려 DNF 입니다.
  2. ★ ROI 좌표 복원. 2025 traffic_sign_detector 는 ROI 로 크롭해 추론하고
     bbox 를 ROI 기준 좌표 그대로 발행했습니다. 융합 노드가 그 bbox 로
     LiDAR 점을 매칭하므로 ROI 를 쓰는 순간 3D 위치가 통째로 틀렸습니다.
  3. ★ 카메라 1대 = 노드 1개. 2025 는 한 노드에서 카메라 3대를 순차 처리해
     지연이 3배였습니다. launch 에서 인스턴스를 여러 개 띄우세요.
  4. ★ 클래스는 모델이 분류한 id 를 그대로 씁니다. 어느 카메라가 보았는지로
     색을 덮어쓰지 않습니다 (2025 융합 노드가 그렇게 했습니다).
  5. 검출이 없어도 빈 Detection2DArray 를 매 주기 발행합니다.
  6. 클래스 매핑을 config/classes.yaml 과 대조하고 다르면 종료합니다.

파라미터
  ~section        classes.yaml 의 섹션명: "cone" 또는 "delivery_sign"
  ~weights        .pt 또는 .engine 경로 (.engine 은 CUDA 필수)
  ~imgsz          추론 해상도. 기본값은 classes.yaml 섹션의 imgsz
  ~image_topic    입력
  ~detections_topic 출력 (vision_msgs/Detection2DArray)
  ~frame_id       Detection2DArray.header.frame_id (카메라 optical frame 권장)
  ~roi            [x1,y1,x2,y2] 원본 기준. 비우면 전체
"""

import os
import re
import sys

import numpy as np
import cv2
def _add_utils_to_path():
    """utils/ 를 import 경로에 추가합니다.

    catkin_install_python 이 스크립트를 devel/lib/<pkg>/ 로 복사하므로
    __file__ 기준 "../utils" 는 그곳에서 존재하지 않습니다.
    소스 트리에서 실행할 때는 상대경로, 그 외에는 rospkg 로 찾습니다."""
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "..", "utils")
    if not os.path.isdir(cand):
        import rospkg
        cand = os.path.join(
            rospkg.RosPack().get_path("mando_vision_2026"), "utils")
    if cand not in sys.path:
        sys.path.insert(0, cand)


_add_utils_to_path()

import rospy
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import (Detection2DArray, Detection2D,
                             BoundingBox2D, ObjectHypothesisWithPose)

_HERE = os.path.dirname(os.path.abspath(__file__))

import class_map                                              # noqa: E402
from infer_loop import LatestFrame, Throttle                  # noqa: E402

PALETTE = [(255, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 255),
           (0, 165, 255), (255, 255, 0), (128, 0, 255)]


class ObjectDetector(object):
    def __init__(self):
        rospy.init_node("object_detector")

        self.section = rospy.get_param("~section", "cone")
        self.conf_th = float(rospy.get_param("~confidence", 0.5))
        self.iou_th = float(rospy.get_param("~iou", 0.45))
        self.infer_hz = float(rospy.get_param("~infer_hz", 10.0))
        self.viz_hz = float(rospy.get_param("~viz_hz", 10.0))
        self.frame_id = rospy.get_param("~frame_id", "cam_front_optical_frame")
        self.roi = rospy.get_param("~roi", [])
        if self.roi and len(self.roi) != 4:
            rospy.logfatal("~roi 는 [x1,y1,x2,y2] 4개여야 합니다: %s", self.roi)
            raise SystemExit(1)

        weights = rospy.get_param("~weights", "")
        if not weights or not os.path.exists(weights):
            rospy.logfatal("가중치를 찾을 수 없습니다: %r", weights)
            raise SystemExit(1)

        # ★ 런타임 pip 설치 없음. 없으면 그냥 죽습니다.
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as e:
            rospy.logfatal("torch/ultralytics 임포트 실패: %s", e)
            rospy.logfatal("실행 전에 설치하세요. 런타임 설치는 하지 않습니다.")
            raise SystemExit(1)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        ext = os.path.splitext(weights)[1].lower()
        if ext == ".engine" and self.device == "cpu":
            # ★ 엔진은 GPU 전용입니다. CPU 폴백으로 조용히 넘어가지 않습니다.
            rospy.logfatal("[%s] %s 는 TensorRT 엔진인데 CUDA 를 쓸 수 없습니다. 종료합니다.",
                           self.section, os.path.basename(weights))
            raise SystemExit(1)
        rospy.loginfo("[%s] 모델 로드: %s (device=%s)",
                      self.section, weights, self.device)
        # ★ task 명시 — 엔진 메타데이터가 불완전하면 task 추론에 실패합니다
        self.model = YOLO(weights, task="detect")
        if ext == ".pt":
            # .engine/.onnx 는 PyTorch 모듈이 아니라 .to() 가 TypeError 로 거부합니다.
            # device 는 predict() 에 직접 넘깁니다.
            self.model.to(self.device)

        sec = class_map.load_and_verify(self.section, self.model.names)
        self.names = sec["names"]
        # ★ TensorRT 엔진은 export 시점 imgsz 로 고정됩니다. rosparam > classes.yaml
        self.imgsz = int(rospy.get_param("~imgsz", sec.get("imgsz", 640)))
        # 대회 당일 .pt 로 돌고 있는 걸 모르는 상황을 막으려고 포맷을 남깁니다
        rospy.loginfo("[%s] %s (%s, imgsz=%d, device=%s)", self.section,
                      os.path.basename(weights),
                      {".engine": "TensorRT", ".pt": "PyTorch",
                       ".onnx": "ONNX"}.get(ext, ext or "?"),
                      self.imgsz, self.device)

        topic = rospy.get_param("~image_topic",
                                "/cam_left/image_raw/compressed")
        out_topic = rospy.get_param("~detections_topic",
                                    "/detect/%s" % self.section)

        self.frame = LatestFrame(topic)
        self.pub = rospy.Publisher(out_topic, Detection2DArray, queue_size=1)
        # ★ rstrip 은 '접미사'가 아니라 '문자 집합'을 제거합니다.
        #   "/perception/cone/detections".rstrip("/detections")
        #     -> "/percep"        (뒤에서부터 /,d,e,t,c,i,o,n,s 를 계속 깎음)
        #   그래서 시각화 토픽이 "/percep/viz/compressed" 가 됐습니다.
        viz_base = re.sub(r"/detections$", "", out_topic)
        self.viz_pub = rospy.Publisher(
            "%s/viz/compressed" % viz_base,
            CompressedImage, queue_size=1)
        self.viz_throttle = Throttle(self.viz_hz)

        rospy.Timer(rospy.Duration(1.0 / self.infer_hz), self.step)
        rospy.loginfo("[%s] 준비 완료 → %s (%.0f Hz)",
                      self.section, out_topic, self.infer_hz)

    def crop(self, img):
        if not self.roi:
            return img, (0, 0)
        h, w = img.shape[:2]
        x1 = max(0, min(int(self.roi[0]), w - 1))
        y1 = max(0, min(int(self.roi[1]), h - 1))
        x2 = max(x1 + 1, min(int(self.roi[2]), w))
        y2 = max(y1 + 1, min(int(self.roi[3]), h))
        return img[y1:y2, x1:x2], (x1, y1)

    def step(self, _evt):
        arr = Detection2DArray()
        arr.header.stamp = rospy.Time.now()
        arr.header.frame_id = self.frame_id

        img, header = self.frame.take()
        if img is None:
            self.pub.publish(arr)            # ★ 빈 배열이라도 매 주기 발행
            return
        if header is not None and header.stamp != rospy.Time(0):
            arr.header.stamp = header.stamp  # 카메라 타임스탬프 보존 (융합용)

        roi_img, (ox, oy) = self.crop(img)
        res = self.model.predict(roi_img, imgsz=self.imgsz, conf=self.conf_th,
                                 iou=self.iou_th, device=self.device,
                                 verbose=False)[0]

        drawn = []
        if res.boxes is not None:
            for b in res.boxes:
                cid = int(b.cls[0])
                if cid not in self.names:
                    rospy.logwarn_throttle(5.0, "매핑에 없는 class id=%d", cid)
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                # ★ ROI 오프셋을 더해 원본 이미지 좌표로 복원
                x1 += ox; x2 += ox; y1 += oy; y2 += oy

                d = Detection2D()
                d.header = arr.header
                bb = BoundingBox2D()
                bb.center.x = (x1 + x2) / 2.0
                bb.center.y = (y1 + y2) / 2.0
                bb.size_x = x2 - x1
                bb.size_y = y2 - y1
                d.bbox = bb

                h = ObjectHypothesisWithPose()
                h.id = cid                        # ★ 모델이 분류한 id 그대로
                h.score = float(b.conf[0])
                d.results.append(h)
                arr.detections.append(d)
                drawn.append((x1, y1, x2, y2, cid, h.score))

        self.pub.publish(arr)

        if self.viz_pub.get_num_connections() > 0 and self.viz_throttle.ready():
            self.publish_viz(img, drawn)

    def publish_viz(self, img, drawn):
        vis = img.copy()
        if self.roi:
            cv2.rectangle(vis, (int(self.roi[0]), int(self.roi[1])),
                          (int(self.roi[2]), int(self.roi[3])), (255, 0, 255), 2)
        for x1, y1, x2, y2, cid, score in drawn:
            c = PALETTE[cid % len(PALETTE)]
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            cv2.putText(vis, "%s %.2f" % (self.names[cid], score),
                        (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, c, 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        m = CompressedImage()
        m.header.stamp = rospy.Time.now()
        m.format = "jpeg"
        m.data = buf.tobytes()
        self.viz_pub.publish(m)


if __name__ == "__main__":
    try:
        ObjectDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
