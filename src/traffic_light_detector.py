#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traffic_light_detector.py — 신호등 관측 노드 (2026)

2025 대비 바뀐 점
  1. GO/STOP 판단을 하지 않습니다. 관측만 발행합니다.
     직진인지 좌회전인지는 경로가 아는 정보지 카메라가 아는 정보가 아닙니다.
     (프로젝트 원칙: 비전 → 시스템 단방향)
  2. ROI 크롭이 실제로 구현되어 있습니다. 2025 는 launch 로 roi 를 넘겼지만
     코드가 읽지도 쓰지도 않았습니다. 좌표는 원본 기준으로 복원해 발행합니다.
  3. 프레임당 1회만 집계 + N프레임 다수결. 2025 는 박스마다 Bool 을 쏴서
     마지막 박스가 이기는 레이스였습니다.
  4. 검출이 없어도 매 주기 발행합니다. 2025 는 검출이 있을 때만 쏴서
     다운스트림이 마지막 값을 영원히 붙들었습니다.
  5. 클래스 매핑을 config/classes.yaml 과 대조하고, 다르면 즉시 종료합니다.
  6. 콜백에서 추론하지 않습니다. 최신 프레임만 잡아 타이머에서 돌립니다.

발행: /perception/traffic_light  (std_msgs/String, JSON)
  {"detected": true, "label": "green", "conf": 0.93,
   "bbox": [x1,y1,x2,y2],           # 원본 이미지 좌표
   "votes": 4, "window": 5, "stamp": 1234567.89}
"""

import os
import sys
from collections import deque, Counter

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

import class_map                                    # noqa: E402
from infer_loop import LatestFrame, HeartbeatPublisher, Throttle   # noqa: E402

COLOR = {
    "green":      (0, 255, 0),
    "green_left": (0, 255, 128),
    "left":       (0, 200, 255),
    "red":        (0, 0, 255),
    "yellow":     (0, 255, 255),
}


class TrafficLightDetector(object):
    def __init__(self):
        rospy.init_node("traffic_light_detector")

        self.conf_th = rospy.get_param("~confidence", 0.5)
        self.infer_hz = rospy.get_param("~infer_hz", 10.0)
        self.vote_win = int(rospy.get_param("~vote_window", 5))
        self.vote_min = int(rospy.get_param("~vote_min", 3))
        self.viz_hz = rospy.get_param("~viz_hz", 5.0)
        topic = rospy.get_param("~image_topic", "/cam_front/color/image_raw/compressed")

        # ROI: [x1, y1, x2, y2] 원본 이미지 기준. 빈 리스트면 전체.
        # ★ 신호등은 작은 객체입니다. 전체를 640 으로 줄이면 픽셀이 날아갑니다.
        #   소실점 주변을 원본 해상도 그대로 크롭해 넣는 편이 훨씬 낫습니다.
        self.roi = rospy.get_param("~roi", [])
        if self.roi and len(self.roi) != 4:
            rospy.logfatal("~roi 는 [x1,y1,x2,y2] 4개여야 합니다: %s", self.roi)
            raise SystemExit(1)

        weights = rospy.get_param("~weights", "")
        if not weights or not os.path.exists(weights):
            rospy.logfatal("가중치를 찾을 수 없습니다: %r", weights)
            raise SystemExit(1)

        # ── 모델 로드 ────────────────────────────────────────────────
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as e:
            rospy.logfatal("torch/ultralytics 임포트 실패: %s", e)
            raise SystemExit(1)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            rospy.logwarn("CUDA 를 쓸 수 없습니다. CPU 로 동작합니다 (매우 느림)")
        rospy.loginfo("모델 로드: %s  (device=%s)", weights, self.device)
        self.model = YOLO(weights)
        self.model.to(self.device)

        # ── ★ 클래스 매핑 대조. 어긋나면 여기서 죽습니다 ─────────────
        sec = class_map.load_and_verify("traffic_light", self.model.names)
        self.names = sec["names"]
        rospy.loginfo("go_straight=%s  go_left=%s",
                      sec.get("go_straight"), sec.get("go_left"))

        self.frame = LatestFrame(topic)
        self.out = HeartbeatPublisher("/perception/traffic_light")
        self.viz_pub = rospy.Publisher(
            "/perception/traffic_light/viz/compressed", CompressedImage, queue_size=1)
        self.viz_throttle = Throttle(self.viz_hz)

        self.votes = deque(maxlen=self.vote_win)
        self.last_box = None

        rospy.Timer(rospy.Duration(1.0 / self.infer_hz), self.step)
        rospy.Timer(rospy.Duration(5.0), self.watchdog)
        rospy.loginfo("traffic_light_detector 준비 완료 (%.0f Hz)", self.infer_hz)

    # ── ROI ──────────────────────────────────────────────────────────
    def crop(self, img):
        if not self.roi:
            return img, (0, 0)
        h, w = img.shape[:2]
        x1 = max(0, min(int(self.roi[0]), w - 1))
        y1 = max(0, min(int(self.roi[1]), h - 1))
        x2 = max(x1 + 1, min(int(self.roi[2]), w))
        y2 = max(y1 + 1, min(int(self.roi[3]), h))
        return img[y1:y2, x1:x2], (x1, y1)

    # ── 메인 루프 ────────────────────────────────────────────────────
    def step(self, _evt):
        img, header = self.frame.take()
        if img is None:
            self.votes.append(None)          # 프레임 없음도 한 표로 셉니다
            self.emit(None, 0.0, None)
            return

        roi_img, (ox, oy) = self.crop(img)
        res = self.model.predict(roi_img, conf=self.conf_th,
                                 device=self.device, verbose=False)[0]

        label, conf, box = None, 0.0, None
        if res.boxes is not None and len(res.boxes):
            # ★ 프레임당 딱 1개 — 가장 신뢰도 높은 박스만
            b = max(res.boxes, key=lambda x: float(x.conf[0]))
            cid = int(b.cls[0])
            label = self.names.get(cid)
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            box = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]   # 원본 좌표로 복원
            if label is None:
                rospy.logwarn_throttle(5.0, "매핑에 없는 class id=%d", cid)

        self.votes.append(label)
        self.emit(label, conf, box)

        if self.viz_pub.get_num_connections() > 0 and self.viz_throttle.ready():
            self.publish_viz(img, box, label, conf)

    def emit(self, label, conf, box):
        """N프레임 다수결로 안정화한 값을 발행합니다. 검출이 없어도 발행합니다."""
        cnt = Counter(self.votes)
        top, n = cnt.most_common(1)[0]
        stable = top if (top is not None and n >= self.vote_min) else None

        if stable is not None and label == stable and box is not None:
            self.last_box = box

        self.out.publish({
            "detected": stable is not None,
            "label": stable,
            "conf": round(conf, 3),
            "bbox": self.last_box if stable is not None else None,
            "votes": n,
            "window": self.vote_win,
        })

    def publish_viz(self, img, box, label, conf):
        vis = img.copy()
        if self.roi:
            cv2.rectangle(vis, (int(self.roi[0]), int(self.roi[1])),
                          (int(self.roi[2]), int(self.roi[3])), (255, 0, 255), 2)
        if box is not None:
            c = COLOR.get(label, (255, 255, 255))
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))
            cv2.rectangle(vis, p1, p2, c, 2)
            cv2.putText(vis, "%s %.2f" % (label, conf), (p1[0], p1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        m = CompressedImage()
        m.header.stamp = rospy.Time.now()
        m.format = "jpeg"
        m.data = buf.tobytes()
        self.viz_pub.publish(m)

    def watchdog(self, _evt):
        if self.frame.received == 0:
            rospy.logwarn("카메라 프레임이 아직 한 장도 오지 않았습니다")


if __name__ == "__main__":
    try:
        TrafficLightDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
