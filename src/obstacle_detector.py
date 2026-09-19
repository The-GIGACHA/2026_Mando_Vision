#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obstacle_detector.py — 장애물 + 신호제어차량 지시 (obstacle.pt)

  배치 위치:  mando_vision_2026/src/obstacle_detector.py

발행
  /perception/sign_car        std_msgs/String (JSON) — 차로 지시
  /detect/obstacle            vision_msgs/Detection2DArray — 전체 검출 (내부 배선)
  /perception/obstacle/viz/compressed            — 구독자 있을 때만
  /perception/s_course/right  std_msgs/Bool — S자 코스 첫 T870 이 우측이면 True (매 주기)
  /perception/s_course        std_msgs/String (JSON) — 위 Bool 의 상태·근거

  ★ /perception/s_course/right 는 기본 False 이고, 다수결 커밋 후 우측 True / 좌측 False 로
    고정된다. False 에는 '아직 모름' 과 '좌측 확정' 이 섞여 있으니 구분이 필요하면
    /perception/s_course 의 state(WAITING / VOTING / COMMITTED) 를 볼 것.
    커밋은 노드가 다시 뜰 때까지 유지된다. 규칙은 utils/s_course.py.

  ★ /perception/obstacle 은 더 이상 여기서 발행하지 않습니다.
    ground_markers 가 /detect/obstacle 을 받아 base_link 좌표로 바꿔 발행하는
    유일한 발행자입니다 (launch/obstacle.launch 가 셋을 같이 띄웁니다).
    obstacle.engine 을 두 번 올리지 않으려고 여기 추론을 그대로 내보냅니다.

────────────────────────────────────────────────────────────────────────
/perception/sign_car 페이로드

  {"detected": true, "command": "LEFT", "basis": "go",
   "confidence": "high", "both_seen": true, "conflict": false,
   "votes": 5, "window": 7, "held": false, "held_s": 0.0,
   "panels": {"left_go": 0.71, "left_X": null,
              "right_go": null, "right_X": 0.66},
   "image_stamp": 1234567.89, "age_ms": 78.3, "stamp": 1234567.97}

  ★ command 는 '관측에서 유도한 값'이지 명령이 아닙니다.
    basis 가 "x" 면 ↓ 를 직접 못 보고 X 만 보고 상보 추론한 것이라
    판단팀이 더 보수적으로 처리해야 합니다.
    confidence 와 conflict 를 반드시 같이 보세요.

  ★ 패널은 점멸합니다 (약 0.67초 켜짐 / 0.67초 꺼짐).
    held=true 면 이번 프레임에 본 것이 아니라 held_s 초 전에 확정한 지시를
    유지 중이라는 뜻입니다 (sign/hold_s 까지, 차체가 사라지면 즉시 해제).
    이때 basis·confidence 는 확정 당시의 근거입니다.

★ 이 노드는 단방향입니다. 구독은 이미지뿐이고, 제어에 관여하지 않습니다.
"""

import os
import sys

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from vision_msgs.msg import (Detection2DArray, Detection2D,
                             BoundingBox2D, ObjectHypothesisWithPose)


def _add_utils_to_path():
    """catkin_install_python 이 스크립트를 devel/lib/<pkg>/ 로 복사하므로
    __file__ 기준 '../utils' 가 거기서는 존재하지 않습니다."""
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "..", "utils")
    if not os.path.isdir(cand):
        import rospkg
        cand = os.path.join(
            rospkg.RosPack().get_path("mando_vision_2026"), "utils")
    if cand not in sys.path:
        sys.path.insert(0, cand)


_add_utils_to_path()

import class_map                                                   # noqa: E402
from infer_loop import LatestFrame, HeartbeatPublisher, Throttle   # noqa: E402
from sign_car import SignCarParams, SignCarVoter, decide, PANELS, HOST  # noqa: E402
from ground import from_params                                     # noqa: E402
from s_course import SCourseJudge                                  # noqa: E402
import mission_gate                                                # noqa: E402

PALETTE = {
    "T870": (60, 220, 60), "kid": (60, 180, 255), "traffic_car": (255, 140, 70),
    "left_go": (80, 255, 80), "right_go": (80, 255, 160),
    "left_X": (60, 60, 255), "right_X": (60, 140, 255),
}


class ObstacleDetector(object):
    def __init__(self):
        rospy.init_node("obstacle_detector")

        topic = rospy.get_param("~image_topic",
                                "/cam_front/color/image_raw/compressed")
        model_path = rospy.get_param("~model", "")
        if not model_path or not os.path.isfile(model_path):
            rospy.logfatal("~model 경로가 잘못됐습니다: %s", model_path)
            raise SystemExit(1)

        # imgsz 는 _load() 에서 classes.yaml 기본값과 합쳐 정합니다.
        self.conf = float(rospy.get_param("~conf", 0.20))
        self.rate_hz = float(rospy.get_param("~rate_hz", 15.0))
        self.viz_hz = float(rospy.get_param("~viz_hz", 10.0))

        self.p = SignCarParams()
        self.p.conf_go = float(rospy.get_param("~sign/conf_go", self.p.conf_go))
        self.p.conf_x = float(rospy.get_param("~sign/conf_x", self.p.conf_x))
        self.p.require_host = bool(rospy.get_param("~sign/require_host", True))
        self.p.host_margin = float(rospy.get_param("~sign/host_margin",
                                                   self.p.host_margin))
        self.p.min_inside = float(rospy.get_param("~sign/min_inside",
                                                  self.p.min_inside))
        self.p.gate_mode = str(rospy.get_param("~sign/gate_mode", self.p.gate_mode))
        self.p.zone_left = tuple(rospy.get_param("~sign/zone_left", self.p.zone_left))
        self.p.zone_right = tuple(rospy.get_param("~sign/zone_right", self.p.zone_right))
        self.p.zone_up = tuple(rospy.get_param("~sign/zone_up", self.p.zone_up))
        self.p.hold_s = float(rospy.get_param("~sign/hold_s", self.p.hold_s))
        self.p.host_lost_s = float(rospy.get_param("~sign/host_lost_s",
                                                   self.p.host_lost_s))
        self.p.vote_n = int(rospy.get_param("~sign/vote_window", self.p.vote_n))
        self.p.vote_k = int(rospy.get_param("~sign/vote_min", self.p.vote_k))

        self._load(model_path)

        self.frame = LatestFrame(topic)
        self.out_sign = HeartbeatPublisher("/perception/sign_car")
        # 전체 검출 — 무엇을 장애물로 쓸지는 ground_markers 의 labels 가 정합니다
        self.frame_id = rospy.get_param("~frame_id",
                                        "cam_front_color_optical_frame")
        self.det_pub = rospy.Publisher(
            rospy.get_param("~detections_topic", "/detect/obstacle"),
            Detection2DArray, queue_size=1)
        self.name_to_id = {v: k for k, v in self.names.items()}
        self.viz_pub = rospy.Publisher(
            "/perception/obstacle/viz/compressed", CompressedImage,
            queue_size=1)
        self.viz_throttle = Throttle(self.viz_hz)
        self.voter = SignCarVoter(self.p)

        # S자 코스: 카메라 기하는 ground_markers 와 같은 from_params 기본값(URDF 실측)을 쓴다.
        # 두 노드가 다른 값을 쓰면 같은 장애물이 토픽마다 다른 좌/우로 나온다.
        self.s_course = SCourseJudge(
            from_params(rospy.get_param, prefix="~ground/"),
            max_x=float(rospy.get_param("~s_course/max_x", 10.0)),
            min_abs_y=float(rospy.get_param("~s_course/min_abs_y", 0.10)),
            vote_window=int(rospy.get_param("~s_course/vote_window", 7)),
            vote_min=int(rospy.get_param("~s_course/vote_min", 5)),
            image_height=int(rospy.get_param("~image_height", 720)))
        self.s_right_pub = rospy.Publisher("/perception/s_course/right", Bool, queue_size=1)
        self.out_s = HeartbeatPublisher("/perception/s_course")

        # 이 모델을 쓰는 미션 구역에서만 추론한다. S자 코스는 판정이 한 번 커밋되면 고정이라
        # 구간 밖 오검출로 미리 커밋되지 않게 '구간 안' 에서만 판정하고, 나가면 초기화한다.
        self.gate = None
        if rospy.get_param("~gate/enable", True):
            self.gate = mission_gate.MissionGate(
                rospy.get_param("~gate/missions",
                                {"S_COURSE": 0.0, "SUDDEN_STOP": 10.0, "SIGNAL_VEHICLE": 10.0}),
                stale_sec=float(rospy.get_param("~gate/stale_sec", 2.0)))
            mission_gate.attach(self.gate, rospy)
        self._gate_mode = None
        self._s_prev_in = None
        self._s_logged = False

        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.step)
        rospy.Timer(rospy.Duration(5.0), self.watchdog)
        rospy.loginfo("obstacle_detector 준비 완료 "
                      "(%.0f Hz, imgsz=%d, ↓임계 %.2f / X임계 %.2f, 차체게이트 %s)",
                      self.rate_hz, self.imgsz, self.p.conf_go, self.p.conf_x,
                      "ON" if self.p.require_host else "OFF")

    # ── 모델 ─────────────────────────────────────────────────────
    def _load(self, path):
        import torch
        from ultralytics import YOLO
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            rospy.logwarn("CUDA 를 못 씁니다. CPU 추론은 실시간이 안 됩니다.")
        # ★ .engine 은 PyTorch 모듈이 아니라 .to() 가 TypeError 로 거부하고,
        #   메타데이터가 불완전하면 task 추론도 실패합니다. device 는 predict()
        #   에 매번 넘기므로 .pt 일 때만 올려둡니다.
        self.net = YOLO(path, task="detect")
        if path.endswith(".pt"):
            self.net.to(dev)
        self.dev = dev
        # ★ classes.yaml 과 '순서까지' 대조합니다. 다르면 여기서 죽습니다.
        #   이름만 대조하면 left_go 와 right_go 의 인덱스가 뒤바뀐 것을
        #   못 잡습니다. 그러면 차로 변경 방향이 정반대로 나갑니다.
        sec = class_map.load_and_verify("obstacle", self.net.names)
        self.names = sec["names"]

        # imgsz: rosparam > classes.yaml > 960
        self.imgsz = int(rospy.get_param("~imgsz", sec.get("imgsz", 960)))

        missing = (set(PANELS) | {HOST}) - set(self.names.values())
        if missing:
            rospy.logfatal("classes.yaml/모델에 없는 클래스: %s", sorted(missing))
            raise SystemExit(1)

        rospy.loginfo("모델 로드: %s  device=%s  imgsz=%d  클래스=%s",
                      os.path.basename(path), dev, self.imgsz, self.names)

        # 워밍업 — 첫 프레임에서 수백 ms 튀는 것을 막습니다
        self.net.predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8),
                         imgsz=self.imgsz, device=dev, verbose=False)

    # ── 주기 ─────────────────────────────────────────────────────
    def _gate(self, now):
        if self.gate is None:
            return True, {"mode": "disabled", "mission": None, "dist": None}, None
        active, info = self.gate.check(now)
        if info["mode"] != self._gate_mode:
            self._gate_mode = info["mode"]
            rospy.loginfo("미션 게이트: %s (%s, %s m)", info["mode"], info["mission"], info["dist"])
        if info["mode"] in ("no_grm", "no_map"):
            rospy.logwarn_throttle(10.0, "미션 게이트 %s — GRM 구간 정보가 없어 항상 켭니다 %s",
                                   info["mode"], info.get("error") or "")
        return active, info, self.gate.in_zone("S_COURSE", now)

    def step(self, _evt):
        active, ginfo, s_in = self._gate(rospy.Time.now().to_sec())
        if self._s_prev_in is True and s_in is False:
            # S자 코스를 벗어났다 — 다음에 다시 들어오면 새로 판정한다
            self.s_course.reset()
            self._s_logged = False
            rospy.loginfo("S자 코스 구간 이탈 — 판정 초기화")
        self._s_prev_in = s_in

        if not active:
            # 켜질 때 구간 밖에서 쌓인 표·프레임으로 판단하지 않게 버린다
            img, hdr = self.frame.take()
            self.voter = SignCarVoter(self.p)
            # 영상이 들어오는데 추론만 쉬는 것이면 영상 stamp 를 붙인다. stamp 0 을 보내면
            # ground_markers 가 '영상 없음' 으로 보고 제어가 카메라 끊김으로 판단한다
            # (S자 코스 레인 모드가 카메라 끊김이면 꺼진다).
            self.publish_detections([], hdr if img is not None else None)
            self.emit(None, None, None)
            self.emit_s_course(None, ginfo)
            return

        img, hdr = self.frame.take()
        if img is None:
            self.publish_detections([], None)
            self.emit(None, None, None)
            self.emit_s_course(None, ginfo)
            return

        dets = self.infer(img)
        self.publish_detections(dets, hdr)
        obs = decide(dets, self.p)
        cmd, votes = self.voter.update(obs)
        self.emit(dets, obs, (cmd, votes), hdr)
        # GRM 이 없으면(s_in None) 항상 켜기 합의에 따라 판정한다
        self.emit_s_course(dets if s_in is not False else None, ginfo)

        if self.viz_pub.get_num_connections() > 0 and self.viz_throttle.ready():
            self.publish_viz(img, dets, obs, cmd)

    def infer(self, img):
        r = self.net.predict(img, imgsz=self.imgsz, conf=self.conf,
                             device=self.dev, verbose=False)[0]
        out = []
        if r.boxes is None:
            return out
        for b in r.boxes:
            cid = int(b.cls[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            out.append({"name": self.names.get(cid, str(cid)),
                        "conf": float(b.conf[0]),
                        "box": (x1, y1, x2, y2)})
        return out

    # ── 발행 ─────────────────────────────────────────────────────
    def publish_detections(self, dets, hdr):
        """infer() 결과 전체를 Detection2DArray 로. 검출이 없어도 매 주기 발행합니다.

        ★ header.stamp 는 원본 영상 stamp 입니다. 영상이 없는 주기는 stamp 0 인
          빈 배열 — ground_markers 가 '봤는데 없음' 과 '영상 없음' 을 구분합니다."""
        arr = Detection2DArray()
        arr.header.frame_id = self.frame_id
        if hdr is not None:
            arr.header.stamp = hdr.stamp
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            det = Detection2D()
            det.header = arr.header
            det.bbox = BoundingBox2D()
            det.bbox.center.x = (x1 + x2) / 2.0
            det.bbox.center.y = (y1 + y2) / 2.0
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.id = self.name_to_id[d["name"]]     # 클래스 id (classes.yaml 대조 완료)
            hyp.score = d["conf"]
            det.results.append(hyp)
            arr.detections.append(det)
        self.det_pub.publish(arr)

    def emit(self, dets, obs, vote, hdr=None):
        now = rospy.Time.now().to_sec()
        istamp = hdr.stamp.to_sec() if hdr is not None else None
        age = round((now - istamp) * 1000.0, 1) if istamp else None

        # ② 차로 지시
        if obs is None:
            self.out_sign.publish({"detected": False, "command": None,
                                   "basis": None, "confidence": None,
                                   "both_seen": False, "conflict": False,
                                   "votes": 0, "window": self.p.vote_n,
                                   "held": False, "held_s": None,
                                   "panels": {k: None for k in PANELS},
                                   "image_stamp": None, "age_ms": None})
            return

        cmd, votes = vote
        ev = self.voter.best_evidence(cmd) if cmd else None
        src = ev if ev else obs
        self.out_sign.publish({
            "detected": cmd is not None,
            "command": cmd,
            "basis": src["basis"],
            "confidence": src["confidence"],
            "both_seen": src["both_seen"],
            "conflict": src["conflict"] or obs["conflict"],
            "votes": votes, "window": self.p.vote_n,
            "held": self.voter.held, "held_s": self.voter.held_s,
            "host": obs["host"],
            "panels": obs["panels"],
            "dropped": obs["dropped"],
            "image_stamp": istamp, "age_ms": age,
        })

    def emit_s_course(self, dets, gate=None):
        """영상이 없는 주기에도 발행한다. 커밋된 값은 영상이 끊겨도 유지해야 하기 때문이다."""
        st = self.s_course.update(dets) if dets is not None else self.s_course.status()
        st["gate"] = gate
        if st["state"] == "COMMITTED" and not getattr(self, "_s_logged", False):
            self._s_logged = True
            rospy.logwarn("S자 코스 첫 T870 판정 커밋: %s (이후 고정)", st["side"])
        self.s_right_pub.publish(Bool(data=st["right"]))
        self.out_s.publish(st)

    # ── 시각화 ───────────────────────────────────────────────────
    def publish_viz(self, img, dets, obs, cmd):
        vis = img.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            c = PALETTE.get(d["name"], (200, 200, 200))
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
            cv2.putText(vis, "%s %.2f" % (d["name"], d["conf"]),
                        (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, c, 2, cv2.LINE_AA)

        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 54), (0, 0, 0), -1)
        txt = "SIGN CAR: %s" % (cmd or "---")
        col = (0, 255, 0) if cmd else (120, 120, 120)
        if obs and obs["conflict"]:
            col = (0, 165, 255)
        cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    col, 2, cv2.LINE_AA)
        if obs:
            sub = "basis=%s conf=%s both=%s host=%s  %s" % (
                obs["basis"], obs["confidence"], obs["both_seen"], obs["host"],
                " ".join("%s=%.2f" % (k, v)
                         for k, v in obs["panels"].items() if v))
            cv2.putText(vis, sub, (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)

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
            rospy.logwarn("전방 카메라 프레임이 아직 한 장도 오지 않았습니다")


if __name__ == "__main__":
    try:
        ObstacleDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
