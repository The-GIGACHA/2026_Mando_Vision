#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_s_course.py — 신호등 Bool 규칙과 S자 코스 좌/우 판정을 ROS·카메라 없이 검증.

  python3 tools/verify_s_course.py

장애물 bbox 는 지면 좌표를 utils/ground.py 정투영(to_pixel)으로 픽셀에 올려 만든다.
★ 같은 투영식으로 만들고 같은 투영식으로 되읽으므로 '투영식이 실차와 맞는가' 는 검증하지 못한다.
  여기서 보는 것은 최근접 선택·거리 제한·중심 불감대·다수결·커밋 고정 로직뿐이다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

from ground import from_params          # noqa: E402
from s_course import SCourseJudge       # noqa: E402
import traffic_light_go as tlg          # noqa: E402

FAILS = []


def check(cond, msg):
    print("  %s  %s" % ("PASS" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


G = from_params(lambda k, d: d)


def t870(X, Y, width_m=0.30, height_m=0.40, conf=0.8):
    """지면 접지점 (X, Y) 에 선 T870 bbox. 밑변 양 끝을 투영하고 높이는 대충 준다."""
    ul, v = G.to_pixel(X, Y + width_m / 2)
    ur, _ = G.to_pixel(X, Y - width_m / 2)
    h_px = 643.3 * height_m / max(X - 0.725, 0.1)
    return {"name": "T870", "conf": conf, "box": (min(ul, ur), v - h_px, max(ul, ur), v)}


print("[1] 신호등 Bool 규칙")
check(tlg.decide("green", (0, 0, 40, 12)) == (True, "green_lamp"), "green → True")
check(tlg.decide("green_left", (0, 0, 20, 6))[0] is True, "green_left → True (녹색 램프가 있어 크기와 무관)")
check(tlg.decide("red", (0, 0, 200, 60))[0] is False, "red → False")
check(tlg.decide("yellow", (0, 0, 200, 60))[0] is False, "yellow → False")
check(tlg.decide(None, None) == (False, "no_detection"), "미검출 → False")
w_min = tlg.ARROW_MIN_PX / tlg.ARROW_RATIO
check(tlg.decide("left", (0, 0, w_min + 1, 20))[0] is True,
      "left, 하우징 %.1f px (화살표 %.1f px ≥ 12) → True" % (w_min + 1, (w_min + 1) * tlg.ARROW_RATIO))
check(tlg.decide("left", (0, 0, w_min - 1, 20)) == (False, "arrow_too_small"),
      "left, 하우징 %.1f px (화살표 %.1f px < 12) → False" % (w_min - 1, (w_min - 1) * tlg.ARROW_RATIO))
check(tlg.decide("left", None) == (False, "arrow_no_box"), "left 인데 bbox 없음 → False")
print("  참고: left 를 True 로 내려면 하우징 bbox 폭이 %.1f px 이상이어야 한다" % w_min)

print("[2] S자 코스 — 우→좌 배치, 먼 두 번째 장애물이 리스트 앞에 와도 가까운 것을 본다")
J = SCourseJudge(G)
check(J.status()["state"] == "WAITING" and J.right is False, "시작: WAITING, Bool False")
for k in range(4):
    st = J.update([t870(18.0, 0.5), t870(9.0 - 0.2 * k, -0.5)])
check(st["state"] == "VOTING" and J.right is False, "4표: 아직 VOTING, Bool False (%s)" % st["votes_right"])
st = J.update([t870(18.0, 0.5), t870(8.0, -0.5)])
check(st["state"] == "COMMITTED" and st["side"] == "RIGHT" and J.right is True,
      "5표째 커밋 → RIGHT, Bool True")
check(abs(st["nearest"]["y"] + 0.5) < 0.02, "최근접 y %.3f (≈ −0.5)" % st["nearest"]["y"])
for _ in range(20):
    st = J.update([t870(6.0, 0.5)])                  # 첫 장애물을 지나 두 번째(좌측)만 보인다
check(st["side"] == "RIGHT" and J.right is True, "커밋 후 좌측 장애물만 계속 보여도 안 뒤집힌다")
st = J.update(None)
check(J.right is True and st["state"] == "COMMITTED", "영상이 끊겨도 커밋 유지")

print("[3] 좌→우 배치")
J = SCourseJudge(G)
for _ in range(5):
    st = J.update([t870(8.5, 0.5), t870(19.0, -0.5)])
check(st["side"] == "LEFT" and J.right is False and st["state"] == "COMMITTED",
      "LEFT 커밋 → Bool False, state COMMITTED 로 '아직 모름' 과 구분")

print("[4] 거리·중심·잘린 박스는 표로 안 센다")
J = SCourseJudge(G)
for _ in range(10):
    st = J.update([t870(14.0, -0.5)])
check(st["state"] == "WAITING" and J.right is False, "10 m 밖(14 m)은 표가 안 쌓인다")
for _ in range(10):
    st = J.update([t870(7.0, 0.04)])
check(st["state"] == "VOTING" and st["votes_left"] == 0 and st["votes_right"] == 0,
      "중심선 |y| < 0.10 m 는 좌/우 표가 아니다")
J = SCourseJudge(G)
cut = t870(5.0, -0.5)
cut["box"] = (cut["box"][0], cut["box"][1], cut["box"][2], 719.0)
for _ in range(10):
    st = J.update([cut])
check(st["state"] == "WAITING", "밑동이 화면 아래로 잘린 박스(y2=719)는 버린다")

print("[5] 오검출이 섞여도 다수결")
J = SCourseJudge(G)
seq = [-0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5]      # 7프레임 중 우측 5, 좌측 2
for y in seq:
    st = J.update([t870(8.0, y)])
check(st["side"] == "RIGHT", "7 중 우 5 / 좌 2 → RIGHT 커밋")

print()
if FAILS:
    print("실패 %d개" % len(FAILS))
    sys.exit(1)
print("전부 통과")
