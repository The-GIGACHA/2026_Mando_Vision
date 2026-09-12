#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obstacle_detector.py — 장애물 + 신호제어차량 지시 (obstacle.pt)

  배치 위치:  mando_vision_2026/src/obstacle_detector.py

발행
  /perception/obstacle        std_msgs/String (JSON) — 장애물 검출 목록
  /perception/sign_car        std_msgs/String (JSON) — 차로 지시
  /perception/obstacle/viz/compressed            — 구독자 있을 때만

────────────────────────────────────────────────────────────────────────
/perception/sign_car 페이로드

  {"detected": true, "command": "LEFT", "basis": "go",
   "confidence": "high", "both_seen": true, "conflict": false,
   "votes": 5, "window": 7,
   "panels": {"left_go": 0.71, "left_X": null,
              "right_go": null, "right_X": 0.66},
   "image_stamp": 1234567.89, "age_ms": 78.3, "stamp": 1234567.97}

  ★ command 는 '관측에서 유도한 값'이지 명령이 아닙니다.
    basis 가 "x" 면 ↓ 를 직접 못 보고 X 만 보고 상보 추론한 것이라
    판단팀이 더 보수적으로 처리해야 합니다.
    confidence 와 conflict 를 반드시 같이 보세요.

★ 이 노드는 단방향입니다. 구독은 이미지뿐이고, 제어에 관여하지 않습니다.
"""

import os
import sys

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage


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
        self.viz_hz = float(rospy.get_param("~viz_hz", 5.0))

        self.p = SignCarParams()
        self.p.conf_go = float(rospy.get_param("~sign/conf_go", self.p.conf_go))
        self.p.conf_x = float(rospy.get_param("~sign/conf_x", self.p.conf_x))
        self.p.require_host = bool(rospy.get_param("~sign/require_host", True))
        self.p.host_margin = float(rospy.get_param("~sign/host_margin",
                                                   self.p.host_margin))
        self.p.min_inside = float(rospy.get_param("~sign/min_inside",
                                                  self.p.min_inside))
        self.p.vote_n = int(rospy.get_param("~sign/vote_window", self.p.vote_n))
        self.p.vote_k = int(rospy.get_param("~sign/vote_min", self.p.vote_k))

        self._load(model_path)

        self.frame = LatestFrame(topic)
        self.out_obs = HeartbeatPublisher("/perception/obstacle")
        self.out_sign = HeartbeatPublisher("/perception/sign_car")
        self.viz_pub = rospy.Publisher(
            "/perception/obstacle/viz/compressed", CompressedImage,
            queue_size=1)
        self.viz_throttle = Throttle(self.viz_hz)
        self.voter = SignCarVoter(self.p)

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
        self.net = YOLO(path)
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
    def step(self, _evt):
        img, hdr = self.frame.take()
        if img is None:
            self.emit(None, None, None)
            return

        dets = self.infer(img)
        obs = decide(dets, self.p)
        cmd, votes = self.voter.update(obs)
        self.emit(dets, obs, (cmd, votes), hdr)

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
    def emit(self, dets, obs, vote, hdr=None):
        now = rospy.Time.now().to_sec()
        istamp = hdr.stamp.to_sec() if hdr is not None else None
        age = round((now - istamp) * 1000.0, 1) if istamp else None

        # ① 장애물 목록 — 검출이 없어도 매 주기 내보냅니다
        items = []
        for d in (dets or []):
            if d["name"] in ("T870", "kid", HOST):
                x1, y1, x2, y2 = d["box"]
                items.append({"label": d["name"], "conf": round(d["conf"], 3),
                              "box": [int(x1), int(y1), int(x2), int(y2)]})
        self.out_obs.publish({"detected": bool(items), "n": len(items),
                              "items": items,
                              "image_stamp": istamp, "age_ms": age})

        # ② 차로 지시
        if obs is None:
            self.out_sign.publish({"detected": False, "command": None,
                                   "basis": None, "confidence": None,
                                   "both_seen": False, "conflict": False,
                                   "votes": 0, "window": self.p.vote_n,
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
            "host": obs["host"],
            "panels": obs["panels"],
            "dropped": obs["dropped"],
            "image_stamp": istamp, "age_ms": age,
        })

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
