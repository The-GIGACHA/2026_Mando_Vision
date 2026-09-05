#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detection_monitor.py — 현장 디버깅용 터미널 대시보드 (2026)

원본: Mando_Vision/src/detection_monitor.py

2025 대비 바뀐 점 — 두 개 다 치명적이었습니다
  1. ★ 'green' in color  →  화이트리스트 정확 매칭
     "green" in "green_left" 가 True 라서, 좌회전 신호에서 직진 GO 판정이
     나왔습니다.
  2. ★ 신선도(freshness) 검사 추가
     2025 get_action_decision() 은 last_update 를 보지 않았습니다.
     화면에는 TIMEOUT 이라 표시하면서 판정은 10초 전 초록불로 GO 를
     계속 내보냈습니다.
  3. ★ 이 노드는 아무것도 발행하지 않습니다. 표시 전용입니다.
     GO/STOP 판단은 planning 의 몫입니다 (비전 → 시스템 단방향).
     여기 나오는 판정은 "지금 인지 상태라면 planning 이 이렇게 볼 것"이라는
     참고용 미리보기입니다.
"""

import os
import sys
import json
import time
from datetime import datetime

import rospy
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "utils"))
import class_map                                              # noqa: E402

FRESH_SEC = 0.5          # 이보다 오래된 관측은 없는 것으로 취급
G, R, Y, B, D, X = ("\033[92m", "\033[91m", "\033[93m",
                    "\033[94m", "\033[90m", "\033[0m")


class Monitor(object):
    def __init__(self):
        rospy.init_node("detection_monitor")

        sec = class_map.load("traffic_light")
        self.go_straight = set(sec.get("go_straight", []))
        self.go_left = set(sec.get("go_left", []))

        self.tl = None
        self.lane = None
        self.cones = (0, 0.0)
        self.signs = (0, 0.0)
        self.intent = "straight"

        rospy.Subscriber("/perception/traffic_light", String,
                         self._cb_json, callback_args="tl", queue_size=1)
        rospy.Subscriber("/perception/lane", String,
                         self._cb_json, callback_args="lane", queue_size=1)
        rospy.Subscriber("/perception/cone/detections", Detection2DArray,
                         self._cb_cone, queue_size=1)
        rospy.Subscriber("/perception/delivery_sign/detections",
                         Detection2DArray, self._cb_sign, queue_size=1)
        rospy.Subscriber("/planning/intent", String, self._cb_intent,
                         queue_size=1)

        self.loop()

    # ── 콜백 ─────────────────────────────────────────────────────────
    def _cb_json(self, msg, which):
        try:
            setattr(self, which, json.loads(msg.data))
        except Exception:
            pass

    def _cb_cone(self, msg):
        self.cones = (len(msg.detections), time.time())

    def _cb_sign(self, msg):
        self.signs = (len(msg.detections), time.time())

    def _cb_intent(self, msg):
        self.intent = (msg.data or "straight").strip()

    # ── 판정 미리보기 ────────────────────────────────────────────────
    @staticmethod
    def _fresh(payload):
        if not payload or "stamp" not in payload:
            return False
        return (rospy.Time.now().to_sec() - payload["stamp"]) < FRESH_SEC

    def preview(self):
        if not self._fresh(self.tl):
            return "STOP", "관측 없음/신선하지 않음"
        label = self.tl.get("label")
        if label is None:
            return "STOP", "신호 미검출"
        allow = self.go_straight if self.intent == "straight" else self.go_left
        if label in allow:
            return "GO", "%s (%s 허용)" % (label, self.intent)
        return "STOP", "%s 는 %s 화이트리스트에 없음" % (label, self.intent)

    # ── 화면 ─────────────────────────────────────────────────────────
    @staticmethod
    def _badge(payload):
        if not payload or "stamp" not in payload:
            return D + "WAITING" + X
        age = rospy.Time.now().to_sec() - payload["stamp"]
        if age > 3.0:
            return R + "TIMEOUT" + X
        if age > FRESH_SEC:
            return Y + "STALE  " + X
        return G + "LIVE   " + X

    def render(self):
        sys.stdout.write("\033[2J\033[H")
        print("=" * 74)
        print(" 2026 MANDO VISION — perception monitor      %s"
              % datetime.now().strftime("%H:%M:%S"))
        print("=" * 74)

        # 신호등
        print("\n[ TRAFFIC LIGHT ]  %s" % self._badge(self.tl))
        if self.tl:
            lab = self.tl.get("label")
            col = {"green": G, "green_left": G, "red": R,
                   "yellow": Y, "left": B}.get(lab, D)
            print("   label      : %s%s%s" % (col, lab or "none", X))
            print("   confidence : %.2f" % self.tl.get("conf", 0.0))
            print("   votes      : %s / %s"
                  % (self.tl.get("votes"), self.tl.get("window")))

        # 차선
        print("\n[ LANE (YOLOPv2) ]  %s" % self._badge(self.lane))
        if self.lane:
            ms = self.lane.get("infer_ms")
            fps = (1000.0 / ms) if ms else 0.0
            print("   lane px    : %.3f %%" % (self.lane.get("lane_ratio", 0) * 100))
            print("   drivable   : %.1f %%" % (self.lane.get("drivable_ratio", 0) * 100))
            print("   inference  : %s ms  (%.1f FPS)"
                  % (ms, fps) if ms else "   inference  : —")

        # 객체
        for title, (n, t) in (("CONE", self.cones), ("DELIVERY SIGN", self.signs)):
            age = time.time() - t if t else 999
            badge = (G + "LIVE" + X) if age < 1.0 else (R + "STALE" + X)
            print("\n[ %s ]  %s   count = %d" % (title, badge, n))

        # 판정 미리보기
        act, why = self.preview()
        col = G if act == "GO" else R
        print("\n" + "-" * 74)
        print(" planning intent : %s" % self.intent)
        print(" preview         : %s%s%s   (%s)" % (col, act, X, why))
        print("-" * 74)
        print(D + " 표시 전용 노드입니다. 실제 판단은 planning 이 합니다." + X)
        sys.stdout.flush()

    def loop(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.render()
            rate.sleep()


if __name__ == "__main__":
    try:
        Monitor()
    except rospy.ROSInterruptException:
        pass
