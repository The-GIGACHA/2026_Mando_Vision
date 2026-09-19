#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_obstacle_depth.py — 뎁스/지면 투영 대조 규칙을 ROS·카메라 없이 검증.

  python3 tools/verify_obstacle_depth.py

평지 + 세워 둔 상자를 ground.py 와 같은 카메라 모델로 뎁스 영상에 그려 넣고,
bbox 는 상자 앞면 꼭짓점을 투영해 만든다.
★ 같은 카메라 모델로 만들고 같은 모델로 되읽으므로 '실차 기하가 맞는가' 는 검증하지 못한다.
  여기서 보는 것은 뎁스 열 샘플링·대조·대체 규칙과 노면 평면 추정뿐이다.
  실데이터 확인은 tools/bag_obstacle_depth.py 로 한다.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils"))

import numpy as np                                                  # noqa: E402

from ground import Ground, from_params                              # noqa: E402
from obstacle_depth import DepthProjector, fit_ground_plane         # noqa: E402

W, H, WHEEL = 1280, 720, 0.150
FAILS = []


def check(cond, msg):
    print("  %s  %s" % ("PASS" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def pixel(G, x, y, z):
    """base_link 점 -> (u, v)."""
    fwd, down = x - G.x_off, G.h - WHEEL - z
    c, s = math.cos(G.th), math.sin(G.th)
    Z = fwd * c + down * s
    return G.cx + G.fx * (G.y_off - y) / Z, G.cy + G.fy * (down * c - fwd * s) / Z


def render(G, boxes):
    """boxes: [(X 앞면, Y 중심, 폭, 높이)] -> 뎁스 [m] (H×W)."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    p, q = (v - G.cy) / G.fy, (u - G.cx) / G.fx
    c, s = math.cos(G.th), math.sin(G.th)
    with np.errstate(divide="ignore", invalid="ignore"):
        Zg = G.h / (p * c + s)
        depth = np.where(Zg > 0, Zg, np.inf)
        for X, Y, w, hgt in boxes:
            Zf = (X - G.x_off) / (c - p * s)
            yb = G.y_off - q * Zf
            zb = G.h - Zf * (p * c + s) - WHEEL
            hit = (Zf > 0) & (np.abs(yb - Y) <= w / 2) & (zb >= -WHEEL) & (zb <= -WHEEL + hgt)
            depth = np.where(hit & (Zf < depth), Zf, depth)
    depth[~np.isfinite(depth) | (depth > 20.0)] = 0.0
    return depth.astype(np.float32)


def bbox(G, X, Y, w, hgt):
    pts = [pixel(G, X, Y + sy * w / 2, -WHEEL + sz * hgt) for sy in (-1, 1) for sz in (0, 1)]
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    return (max(0, min(us)), max(0, min(vs)), min(W - 1, max(us)), min(H - 1, max(vs)))


G = from_params(lambda k, d: d)
DP = DepthProjector(G)
T870 = (0.30, 0.40)

print("[1] 뎁스가 지면 투영과 맞으면 뎁스 점을 쓴다")
b = (4.0, 0.5) + T870
r = DP.measure(bbox(G, *b), render(G, [b]))
check(r["used"] == "depth" and r["reason"] == "ok", "4 m 좌측 T870 → depth/ok (%s/%s)" % (r["used"], r["reason"]))
check(abs(r["depth"]["x"] - 4.0) < 0.05, "뎁스 x %.3f ≈ 4.0 (앞면)" % r["depth"]["x"])
check(abs(r["ground"]["x"] - 4.0) < 0.05, "지면 x %.3f ≈ 4.0" % r["ground"]["x"])
check(len(r["points"]) == 5 and all(abs(p[1] - 0.5) < 0.16 for p in r["points"]),
      "점 5개, y 가 상자 폭 안 (%s)" % [round(p[1], 2) for p in r["points"]])
check(all(-0.5 <= p[2] <= 2.5 for p in r["points"]), "z 가 제어 높이 필터(-0.5~2.5) 안")

print("[2] 역광 구멍 — 박스 안 뎁스가 비면 지면 투영")
d = render(G, [b])
x1, y1, x2, y2 = [int(v) for v in bbox(G, *b)]
d[y1:y2 + 1, x1:x2 + 1] = 0.0
r = DP.measure(bbox(G, *b), d)
check(r["used"] == "ground" and r["reason"] == "few_valid", "few_valid → ground")
check(len(r["points"]) == 5 and all(abs(p[0] - 4.0) < 0.1 for p in r["points"]), "지면 점 5개 x≈4.0")

print("[3] 뎁스 오인식 — 거리가 틀리면 지면 투영")
r = DP.measure(bbox(G, *b), render(G, [b]) * 1.4)
check(r["used"] == "ground" and r["reason"] == "mismatch",
      "뎁스 1.4배 (%.2f vs 지면 %.2f) → mismatch" % (r["depth"]["x"], r["ground"]["x"]))
r = DP.measure(bbox(G, *b), render(G, [b]) * 1.08)
check(r["used"] == "depth", "뎁스 8%% 오차 (%.2f, 허용 %.2f) 는 받아들인다" % (r["depth"]["x"], r["depth"]["tol"]))

print("[4] 밑동이 잘린 근거리 — 지면 투영이 못 재는 곳을 뎁스로")
# 0.40 m 높이 T870 은 base_link 약 1.91 m 보다 가까우면 화면에서 통째로 사라진다
b = (2.2, -0.3) + T870
bb = bbox(G, *b)
r = DP.measure(bb, render(G, [b]))
check(r["cut"] and bb[3] >= H - 2, "2.2 m 박스 밑변이 화면 끝 (y2=%.0f)" % bb[3])
check(r["used"] == "depth" and r["reason"] == "cut_ok" and abs(r["depth"]["x"] - 2.2) < 0.05,
      "cut_ok → 뎁스 x %.3f ≈ 2.2 (지면 투영은 %.2f 로 뭉개짐)" % (r["depth"]["x"], r["ground"]["x"]))
r = DP.measure(bb, np.full((H, W), 5.0, np.float32))
check(r["used"] == "ground" and r["reason"] == "cut_far", "잘렸는데 뎁스가 5 m → cut_far, 지면(더 가까운 값)")

print("[5] 먼 거리·뎁스 없음")
b = (10.0, 0.4) + T870
r = DP.measure(bbox(G, *b), render(G, [b]))
check(r["used"] == "ground" and r["reason"] == "far", "10 m (depth_max 8 m 밖) → far")
b = (4.0, 0.5) + T870
r = DP.measure(bbox(G, *b), None)
check(r["used"] == "ground" and r["reason"] == "no_depth", "뎁스 프레임 없음 → no_depth")

print("[6] 두 장애물 — 가까운 쪽이 먼 쪽 bbox 를 가려도 각자 자기 거리")
b1, b2 = (3.0, 0.45) + T870, (6.0, -0.45) + T870
d = render(G, [b1, b2])
r1, r2 = DP.measure(bbox(G, *b1), d), DP.measure(bbox(G, *b2), d)
check(r1["used"] == "depth" and abs(r1["depth"]["x"] - 3.0) < 0.05, "3 m 좌 → %.2f" % r1["depth"]["x"])
check(r2["used"] == "depth" and abs(r2["depth"]["x"] - 6.0) < 0.1, "6 m 우 → %.2f" % r2["depth"]["x"])

print("[7] 노면 평면으로 카메라 높이·pitch 재기")
# 실제 카메라가 설정보다 1.75° 더 위를 본다고 둔다 (09-16 bag 에서 설정 -1.0°, 실측 -2.75° 였던 차이)
TRUE_PITCH = math.degrees(G.th) - 1.75
Gt = Ground(G.fx, G.fy, G.cx, G.cy, 0.974, TRUE_PITCH, G.x_off, G.y_off)
f = fit_ground_plane(render(Gt, [(4.0, 0.5) + T870]), G.fx, G.fy, G.cx, G.cy)
check(f is not None and abs(f["h"] - 0.974) < 0.005 and abs(f["pitch_deg"] - TRUE_PITCH) < 0.05,
      "참값 h 0.974 / pitch %.2f° → %.3f / %.2f° (inlier %.0f%%)"
      % (TRUE_PITCH, f["h"], f["pitch_deg"], f["inlier"] * 100))

print("[8] pitch 가 1.75° 틀리면 지면 투영만 짧아진다 — 뎁스는 거의 그대로 (2026-09-16 송도 bag 상황)")
# ★ 합성 상자에서는 1.75° 오차가 7 m 까지 허용오차(15%) 안이라 뎁스를 계속 쓴다.
#   실제 bag 에서 mismatch 가 40% 까지 난 것은 pitch 오차에 콘 경사면·bbox 여유가 겹친 결과로 본다 (추측).
gaps = []
for X in (3.0, 5.0, 7.0):
    b = (X, 0.5) + T870
    r = DP.measure(bbox(Gt, *b), render(Gt, [b]))
    gaps.append(r["depth"]["x"] - r["ground"]["x"])
    print("        %.0f m: 지면 %.2f  뎁스 %.2f  허용 %.2f  → %s/%s"
          % (X, r["ground"]["x"], r["depth"]["x"], r["depth"]["tol"], r["used"], r["reason"]))
check(gaps[0] < gaps[1] < gaps[2] and gaps[2] > 0.8,
      "뎁스-지면 차이가 거리 따라 커진다 (%s m)" % ", ".join("%.2f" % g for g in gaps))

print()
if FAILS:
    print("실패 %d개" % len(FAILS))
    sys.exit(1)
print("전부 통과")
