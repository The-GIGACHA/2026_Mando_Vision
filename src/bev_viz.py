#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bev_viz.py — S코스 좌/우 판정을 눈으로 확인하는 BEV 시각화 (표시 전용)

  rosrun mando_vision_2026 bev_viz.py
  rqt_image_view   ->  /perception/bev/compressed

★ 이 노드는 모델을 올리지 않습니다. VRAM 0, 순수 CPU 기하 연산입니다.
  /perception/obstacle 의 bbox 를 받아 접지점만 지면으로 투영합니다.

★ 왜 영상 전체를 워핑하지 않는가
  호모그래피는 '지면 위의 점'만 올바르게 옮깁니다. T870 은 지면에서 솟은
  입체라, 전체를 워핑하면 카메라 반대 방향으로 길게 늘어진 줄무늬가 되고
  그 줄무늬 중심은 실제 위치가 아닙니다. bbox 하단 모서리 중심(= 지면과
  닿는 선)만이 진짜 지면 위의 점이고, 그 한 점만 변환하면 충분합니다.
  전체 워핑은 --warp 로 켤 수 있지만 배경 확인용일 뿐 판정 근거가 아닙니다.

★ 아무도 안 보면 아무 일도 하지 않습니다
  get_num_connections() 게이트 + Throttle. 주행 중에는 viz_hz:=0 으로 끄십시오.
"""
import json
import math
import os
import sys

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.ground import from_params as ground_from_params  # noqa: E402
from utils.infer_loop import LatestFrame, Throttle  # noqa: E402


class BevCanvas(object):
    """base_link 기준 탑다운 캔버스. 위가 전방, 왼쪽이 차량 좌측."""

    def __init__(self, x_max, y_half, px_per_m):
        self.x_max, self.y_half, self.s = x_max, y_half, px_per_m
        self.W = int(2 * y_half * px_per_m)
        self.H = int(x_max * px_per_m)

    def xy2px(self, X, Y):
        # Y 좌측+ 인데 화면은 오른쪽이 +u 이므로 좌우를 뒤집습니다.
        return int((self.y_half - Y) * self.s), int((self.x_max - X) * self.s)

    def blank(self):
        img = np.full((self.H, self.W, 3), 28, np.uint8)
        for X in range(0, int(self.x_max) + 1, 5):
            _, py = self.xy2px(X, 0)
            cv2.line(img, (0, py), (self.W, py), (60, 60, 60), 1)
            cv2.putText(img, "%dm" % X, (4, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)
        for Y in np.arange(-self.y_half, self.y_half + .01, 1.0):
            px, _ = self.xy2px(0, Y)
            cv2.line(img, (px, 0), (px, self.H), (60, 60, 60), 1)
        # 차체 중심선 = 좌/우 판정 경계
        px, _ = self.xy2px(0, 0)
        cv2.line(img, (px, 0), (px, self.H), (0, 190, 255), 2)
        # 차량 (뒷차축 원점)
        p0 = self.xy2px(0.0, 0.35)
        p1 = self.xy2px(0.9, -0.35)
        cv2.rectangle(img, p0, p1, (0, 160, 220), 2)
        return img


class Node(object):
    def __init__(self):
        g = rospy.get_param
        # obstacle_detector 와 같은 모듈을 씁니다. 투영식이 두 벌이면
        # 한쪽만 고쳤을 때 화면과 발행값이 어긋납니다.
        self.G = ground_from_params(g)
        self.canvas = BevCanvas(g("~x_max", 30.0), g("~y_half", 6.0),
                                g("~px_per_m", 18.0))
        self.labels = set(g("~labels", ["T870"]))
        self.decide_max = g("~decide_range", 10.0)   # 이 안쪽만 판정에 씀
        self.warp = g("~warp", False)
        self.thr = Throttle(g("~viz_hz", 5.0))

        self.frame = LatestFrame(g("~image", "/cam_front/color/image_raw/compressed"))
        self.det = None
        rospy.Subscriber(g("~obstacle", "/perception/obstacle"),
                         String, self._cb, queue_size=1)
        self.pub = rospy.Publisher("/perception/bev/compressed",
                                   CompressedImage, queue_size=1)
        self._Hinv = None
        rospy.Timer(rospy.Duration(0.05), self.step)
        rospy.loginfo("bev_viz: h=%.3f pitch=%.1f y_off=%.3f warp=%s",
                      self.G.h, math.degrees(self.G.th), self.G.y_off, self.warp)

    def _cb(self, msg):
        try:
            self.det = json.loads(msg.data)
        except ValueError:
            self.det = None

    # ── 전체 워핑 (배경 확인용, 기본 off) ──────────────────────────────
    def _warp_map(self, shape):
        """지면->픽셀 대응표. 한 번만 만들고 재사용합니다."""
        if self._Hinv is not None:
            return self._Hinv
        c, G = self.canvas, self.G
        ys, xs = np.mgrid[0:c.H, 0:c.W]
        X = c.x_max - ys / c.s - G.x_off
        Y = c.y_half - xs / c.s - G.y_off
        den = X * math.cos(G.th) + G.h * math.sin(G.th)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = G.cx - G.fx * Y / den
            v = G.cy + G.fy * (G.h * math.cos(G.th) - X * math.sin(G.th)) / den
        bad = (den <= 0.05) | (X <= 0)
        u[bad], v[bad] = -1, -1
        self._Hinv = (u.astype(np.float32), v.astype(np.float32))
        return self._Hinv

    # ── 본체 ───────────────────────────────────────────────────────────
    def step(self, _):
        # 아무도 안 보면 인코딩도 변환도 하지 않습니다.
        if self.pub.get_num_connections() == 0 or not self.thr.ready():
            return
        got = self.frame.get()
        if got is None:
            return
        im, hdr = got

        if self.warp:
            mu, mv = self._warp_map(im.shape)
            bev = cv2.remap(im, mu, mv, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(28, 28, 28))
            bev = (0.55 * bev + 0.45 * self.canvas.blank()).astype(np.uint8)
        else:
            bev = self.canvas.blank()

        votes = {"L": 0, "R": 0}
        rows = []
        for it in ((self.det or {}).get("items") or []):
            if it.get("label") not in self.labels:
                continue
            # 발행된 값을 그대로 그립니다. 화면과 다운스트림이 같은 숫자를
            # 보게 하려는 것이고, 여기서 다시 계산하면 둘이 갈라집니다.
            r = it.get("ground") or self.G.judge_box(it["box"])
            if r is None:
                continue
            X, Y, side, nsig = r["x"], r["y"], r["side"], r["sigma"]
            usable = X <= self.decide_max
            if usable:
                votes[side] += 1

            col = (90, 220, 90) if usable else (120, 120, 120)
            px, py = self.canvas.xy2px(X, Y)
            cv2.circle(bev, (px, py), 7, col, -1)
            cv2.line(bev, self.canvas.xy2px(X, 0), (px, py), col, 1)
            cv2.putText(bev, "%s %.2fm %.0fs" % (side, Y, nsig),
                        (px + 10, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            rows.append((it["label"], X, Y, side, nsig, it.get("conf")))

        # 판정 범위 경계선
        _, py = self.canvas.xy2px(self.decide_max, 0)
        cv2.line(bev, (0, py), (self.canvas.W, py), (0, 120, 200), 1)
        cv2.putText(bev, "decide <= %.0fm" % self.decide_max, (6, py + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 220), 1)

        tot = votes["L"] + votes["R"]
        head = "no T870 in range" if tot == 0 else \
               "L %d / R %d" % (votes["L"], votes["R"])
        cv2.putText(bev, head, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (240, 240, 240), 2)
        for i, (lb, X, Y, s, ns, cf) in enumerate(rows[:4]):
            cv2.putText(bev, "%s X=%.1f Y=%+.2f %s %.0fsig c=%.2f"
                        % (lb, X, Y, s, ns, cf or 0),
                        (8, 44 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1)

        age = (self.det or {}).get("age_ms")
        if age is not None and age > 300:
            cv2.putText(bev, "STALE %.0fms" % age, (8, self.canvas.H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 240), 2)

        msg = CompressedImage()
        msg.header = hdr if hdr is not None else rospy.Header()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode(".jpg", bev,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).tobytes()
        self.pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("bev_viz")
    Node()
    rospy.spin()
