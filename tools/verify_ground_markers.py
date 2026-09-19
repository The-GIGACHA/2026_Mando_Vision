#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_ground_markers.py — ground_markers 계산을 ROS 마스터 없이 검증

  python3 tools/verify_ground_markers.py

가짜 Detection2DArray (콘 3 / 4 / 5 m) 를 utils/obstacle_ground.py 에 넣고
좌표·높이 복원·x_err·트래킹·투표·소스 끊김·페이로드 키 집합을 확인합니다.
노드(src/ground_markers.py)는 이 모듈을 그대로 부르고 발행만 합니다.

★ 가짜 픽셀은 ground.py 역변환이 아니라 회전행렬로 쓴 '독립' 핀홀 투영으로
  만듭니다. 같은 식으로 만들고 같은 식으로 풀면 틀려도 통과하기 때문입니다.
"""
import json
import math
import os
import random
import re
import sys
from types import SimpleNamespace as NS

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "utils"))

from ground import from_params                                  # noqa: E402
from obstacle_ground import GroundFusion, Projector, from_detection2d  # noqa: E402

try:
    from vision_msgs.msg import (Detection2DArray, Detection2D,
                                 ObjectHypothesisWithPose)
    MSG = "vision_msgs 실제 메시지 클래스"
except ImportError:                       # ROS 가 아예 없는 PC
    MSG = "SimpleNamespace 대역 (vision_msgs 없음)"

    class Detection2DArray(object):
        def __init__(self):
            self.header = NS(stamp=None, frame_id="")
            self.detections = []

    class Detection2D(object):
        def __init__(self):
            self.bbox = NS(center=NS(x=0.0, y=0.0, theta=0.0), size_x=0.0, size_y=0.0)
            self.results = []

    class ObjectHypothesisWithPose(object):
        def __init__(self):
            self.id, self.score = 0, 0.0

W, H = 1280, 720
CONE_W, CONE_H = 0.20, 0.45
FAILS = []


def check(ok, what):
    print("   [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        FAILS.append(what)


def names_of(section):
    with open(os.path.join(PKG, "config", "classes.yaml"), encoding="utf-8") as f:
        sec = yaml.safe_load(f)[section]
    return {int(k): re.sub(r"^\d+_", "", str(v)).strip()
            for k, v in sec["names"].items()}


# ── 독립 핀홀 ────────────────────────────────────────────────────────
def pinhole(G, X, Y, Z):
    """base_link (X 전방, Y 좌+, Z 위) -> 픽셀. pitch th 는 아래를 보면 +."""
    dx, dy, dz = X - G.x_off, Y - G.y_off, Z - G.h
    c, s = math.cos(G.th), math.sin(G.th)
    zc = dx * c - dz * s                  # 광축 방향 (cos, 0, -sin)
    yc = -dx * s - dz * c                 # 영상 아래 (-sin, 0, -cos)
    xc = -dy                              # 영상 오른쪽 = 차량 우측
    return G.cx + G.fx * xc / zc, G.cy + G.fy * yc / zc


def cone_box(G, X, Y, w=CONE_W, h=CONE_H, clip=True):
    """거리 X 에 선 폭 w, 높이 h 의 수직판 -> bbox"""
    pts = [pinhole(G, X, Y + sg * w / 2.0, z) for sg in (1, -1) for z in (0.0, h)]
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    x1, y1, x2, y2 = min(us), min(vs), max(us), max(vs)
    if clip:
        x1, x2 = max(0.0, x1), min(W - 1.0, x2)
        y1, y2 = max(0.0, y1), min(H - 1.0, y2)
    return x1, y1, x2, y2


def make_array(stamp, dets):
    """dets: [(class_id, score, (x1, y1, x2, y2))]"""
    arr = Detection2DArray()
    arr.header.stamp = stamp
    for cid, score, (x1, y1, x2, y2) in dets:
        d = Detection2D()
        d.bbox.center.x, d.bbox.center.y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        d.bbox.size_x, d.bbox.size_y = x2 - x1, y2 - y1
        hyp = ObjectHypothesisWithPose()
        hyp.id, hyp.score = cid, score
        d.results.append(hyp)
        arr.detections.append(d)
    return arr


def main():
    G = from_params(lambda k, d: d)
    P = Projector(G, 0.5)
    print("입력 메시지: %s" % MSG)
    print("카메라: h=%.3f m  pitch=%.1f°  x_off=%.3f  y_off=%.3f"
          % (G.h, math.degrees(G.th), G.x_off, G.y_off))

    # ── 0. 독립 핀홀이 ground.py 와 같은 카메라인지 ───────────────────
    print("\n[0] 핀홀 대조 (지면점)")
    err = max(abs(a - b) for X in (3, 5, 9) for Y in (-1, 0, 1)
              for a, b in zip(pinhole(G, X, Y, 0.0), G.to_pixel(X, Y)))
    check(err < 1e-9, "지면점 픽셀 일치 (최대 차 %.1e px)" % err)

    # ── 1. 좌표 · 높이 · x_err ────────────────────────────────────────
    print("\n[1] 콘 3 / 4 / 5 m — 좌표·높이 복원 (연속 픽셀)")
    print("   실제(X, Y)        x       y       z     width  x_err  box")
    truth = [(3.0, 0.50), (4.0, -0.30), (5.0, 0.00)]
    for X, Y in truth:
        b = cone_box(G, X, Y)
        g = P.project(*b)
        print("   (%.1f, %+.2f)   %.4f %+.4f %.4f %.4f %.3f  [%d %d %d %d]"
              % (X, Y, g["x"], g["y"], g["z"], g["width"], g["x_err"], *map(int, b)))
        # 화면 중심에서 벗어난 물체는 bbox 의 좌/우 끝 중 하나가 '윗변' 모서리에서
        # 나옵니다 (원근). 실제 검출 박스도 같아서 y·width 가 mm 단위로 어긋납니다.
        check(abs(g["x"] - X) < 1e-6, "%.0f m: x 복원 (오차 %.1e)" % (X, abs(g["x"] - X)))
        check(abs(g["y"] - Y) < 0.005 and abs(g["width"] - CONE_W) < 0.005,
              "%.0f m: y, width 오차 < 5 mm (%.1f / %.1f mm)"
              % (X, 1000 * abs(g["y"] - Y), 1000 * abs(g["width"] - CONE_W)))
        check(abs(g["z"] - CONE_H) < 1e-6, "%.0f m: z=%.4f (오차 %.1e)"
              % (X, g["z"], abs(g["z"] - CONE_H)))

    b5 = cone_box(G, 5.0, 0.0)
    top_as_ground = G.to_ground((b5[0] + b5[2]) / 2, b5[1])[0]
    print("   참고: 5 m 콘의 윗변을 지면 투영하면 %.2f m (그래서 윗변은 z 에만 씁니다)"
          % top_as_ground)
    check(abs(top_as_ground - 8.50) < 0.05, "윗변 지면 투영 = 8.50 m 재현")

    print("\n   x_err = pitch ±0.5° 로 흔든 x 의 폭. 독립 계산과 대조")
    print("     (수평거리 d, 내려본 각 a=atan(h/d) -> h/tan(a-0.5°) - h/tan(a+0.5°))")
    print("   실측표 'pitch 1° 오차' 는 +1° 한쪽으로만 틀어진 값이라 같이 적습니다")
    ref = {3.0: 0.10, 5.0: 0.31, 7.0: 0.64, 10.0: 1.31}
    for X, r in ref.items():
        e = P.project(*cone_box(G, X, 0.0))["x_err"]
        d = X - G.x_off
        a = math.atan2(G.h, d)
        spread = G.h / math.tan(a - math.radians(0.5)) - G.h / math.tan(a + math.radians(0.5))
        one = d - G.h / math.tan(a + math.radians(1.0))
        print("     %4.1f m  x_err=%.3f  독립계산 %.3f   | +1° 한쪽 %.3f  실측표 %.2f"
              % (X, e, spread, one, r))
        check(abs(e - spread) < 1e-9, "%.0f m x_err = 독립 계산" % X)
        check(abs(one - r) < 0.01, "%.0f m 실측표 = +1° 한쪽 오차" % X)

    print("\n[2] 정수 픽셀 (검출기 반올림) 영향")
    for X, Y in truth:
        b = [float(round(v)) for v in cone_box(G, X, Y)]
        g = P.project(*b)
        print("   %.0f m: x=%.3f (%+.3f)  y=%+.3f (%+.3f)  z=%.3f (%+.3f)"
              % (X, g["x"], g["x"] - X, g["y"], g["y"] - Y, g["z"], g["z"] - CONE_H))
        check(abs(g["x"] - X) < 0.03 and abs(g["z"] - CONE_H) < 0.03,
              "%.0f m: 반올림 오차 x, z < 3 cm" % X)

    print("\n[3] 가까워서 밑동이 잘린 콘 (2.0 m)")
    fus = GroundFusion(P, height_expect={"cone_blue": CONE_H})
    b = cone_box(G, 2.0, 0.0)
    pay, warns = fus.step([("/detect/cone", "cone", 1.0,
                            [("cone_blue", 0.9) + tuple(b)])], 1.0)
    it = pay["items"][0]
    print("   box=%s  x=%.2f  z=%s  경고=%s" % (it["box"], it["x"], it["z"], warns))
    check(it["box"][3] == H - 1, "box y2 = %d (이미지 높이-1 → 잘림 식별 가능)" % (H - 1))
    check(abs(it["x"] - 2.70) < 0.02, "x 가 2.70 m 로 뭉개짐 (%.2f)" % it["x"])
    check(not warns, "잘린 박스는 높이 경고에서 제외")

    # ── 4. 트래킹 · 투표 · 소스 끊김 ──────────────────────────────────
    print("\n[4] 트래킹 / 투표 / 라벨 필터 / 소스 끊김  (15 Hz 주기, 콘 소스는 3틱 중 2틱)")
    src = {"/detect/obstacle": ("obstacle", names_of("obstacle"), {"T870", "kid"}),
           "/detect/cone": ("cone", names_of("cone"), {"cone_blue", "cone_yellow"})}
    oid = {v: k for k, v in src["/detect/obstacle"][1].items()}
    cid = {v: k for k, v in src["/detect/cone"][1].items()}
    rng = random.Random(7)
    fus = GroundFusion(P, gate_m=0.8, miss_max=5, vote_window=5, vote_min=3,
                       source_timeout=0.5, height_expect={"cone_blue": 0.45,
                       "cone_yellow": 0.45, "T870": 0.40, "kid": 0.60})
    cones = [("cone_blue", 3.0, 0.50), ("cone_yellow", 4.0, -0.30), ("cone_blue", 5.0, 0.00)]

    def jit(b):
        return tuple(v + rng.uniform(-1.0, 1.0) for v in b)

    hist, cone_frames = [], 0
    for k in range(45):
        now = 100.0 + k / 15.0
        frames = []
        cone_up = (k % 3 != 2) and k < 24          # 24틱부터 콘 소스 끊김
        obst_up = k < 34                           # 34틱부터 전부 끊김
        if cone_up:
            cone_frames += 1
            dets = [(cid[l], 0.9, jit(cone_box(G, X, Y))) for l, X, Y in cones
                    if not (X == 4.0 and cone_frames in (8, 9))]   # 4 m 콘 두 프레임 가림
            frames.append(("/detect/cone", make_array(now - 0.04, dets)))
        if obst_up:
            dets = [(oid["traffic_car"], 0.8, (900, 300, 1100, 420)),   # 라벨 필터 대상
                    (oid["left_go"], 0.7, (950, 320, 980, 350))]
            if k == 6:                                                 # 한 프레임 오검출
                dets.append((oid["kid"], 0.35, jit(cone_box(G, 6.0, 1.0, 0.3, 0.6))))
            if k >= 18:
                dets.append((oid["T870"], 0.85, jit(cone_box(G, 7.0, -0.2, 0.35, 0.40))))
            frames.append(("/detect/obstacle", make_array(now - 0.05, dets)))

        fr = []
        for key, arr in frames:
            section, names, labels = src[key]
            dets = [r for r in (from_detection2d(d, names, labels) for d in arr.detections) if r]
            fr.append((key, section, arr.header.stamp, dets))
        pay, warns = fus.step(fr, now)
        json.dumps(pay)                                     # 직렬화 가능해야 함
        hist.append((k, [f[0][8:] for f in fr], pay, warns))

    for k, srcs, pay, warns in hist:
        s = "  ".join("id%d:%s x=%.2f v%d%s" % (i["id"], i["label"][:6], i["x"], i["votes"],
                                                "" if i["stable"] else "?")
                      for i in pay["items"])
        print("   t%02d %-17s n=%d %s%s" % (k, "+".join(srcs) or "-", pay["n"], s,
                                          "  경고" if warns else ""))

    def track(pay, X):
        return [i for i in pay["items"] if i["label"].startswith("cone") and abs(i["x"] - X) < 0.3]

    ids = {X: set(i["id"] for _, _, p, _ in hist[:24] for i in track(p, X)) for X in (3.0, 4.0, 5.0)}
    print("   콘별 id: %s" % {X: sorted(v) for X, v in ids.items()})
    check(all(len(v) == 1 for v in ids.values()) and
          len(set.union(*ids.values())) == 3, "콘 3개가 끝까지 각자 같은 id")

    v3 = [track(p, 3.0)[0]["votes"] for _, s, p, _ in hist[:24] if "cone" in s]
    st3 = [track(p, 3.0)[0]["stable"] for _, s, p, _ in hist[:24] if "cone" in s]
    print("   3 m 콘 votes (콘 프레임마다): %s" % v3[:8])
    check(v3[:6] == [1, 2, 3, 4, 5, 5], "votes 1→5 누적, window 5 에서 포화")
    check(st3[:4] == [False, False, True, True], "3표부터 stable (그 전에도 발행은 됨)")

    hid = [(k, p) for k, s, p, _ in hist if k < 24 and "cone" in s]
    gone = [k for k, p in hid if not track(p, 4.0)]
    back = [track(p, 4.0)[0] for k, p in hid if k > max(gone)][0]
    print("   4 m 콘: 가려진 틱 %s → 복귀 id=%d votes=%d" % (gone, back["id"], back["votes"]))
    check(len(gone) == 2 and back["id"] in ids[4.0], "가려진 동안 목록에서 빠지고 같은 id 로 복귀")
    check(back["votes"] == 3, "복귀 직후 votes 가 깎여 있음 (창 5칸에 놓친 2번이 남아 3)")

    kid = [(k, i) for k, _, p, _ in hist for i in p["items"] if i["label"] == "kid"]
    print("   오검출 kid: %s" % [(k, i["votes"], i["stable"]) for k, i in kid])
    check(len(kid) == 1 and kid[0][1]["votes"] == 1 and not kid[0][1]["stable"],
          "한 프레임 오검출은 votes=1, stable=false 로 발행되고 다음 프레임에 사라짐")
    check(not any(t.ob["label"] == "kid" for t in fus.tracks), "kid 트랙은 miss_max 후 삭제")

    allab = set(i["label"] for _, _, p, _ in hist for i in p["items"])
    check(not (allab & {"traffic_car", "left_go"}), "labels 밖(traffic_car, left_go)은 제외")

    k30 = hist[30][2]                    # 콘 마지막 프레임(t22) 후 0.53 s
    print("   t30 (콘 소스 0.5 s 넘게 끊김): %s  image_stamp=%.3f"
          % ([(i["label"], i["source"]) for i in k30["items"]], k30["image_stamp"]))
    check(hist[28][2]["n"] == 4, "t28 (끊긴 지 0.4 s) 까지는 콘 항목 유지 — source_timeout 0.5 s")
    check(k30["n"] == 1 and k30["items"][0]["label"] == "T870",
          "콘 소스가 끊겨도 obstacle 소스는 계속 발행")
    t870 = [i for _, _, p, _ in hist for i in p["items"] if i["label"] == "T870"]
    check(abs(t870[-1]["z"] - 0.40) < 0.03, "T870 대역(0.40 m) 높이 복원 z=%.3f" % t870[-1]["z"])

    check(not any(w for *_, w in hist), "정상 캘리브레이션에서 높이 경고 없음")

    # ── 5. 페이로드 키 ────────────────────────────────────────────────
    print("\n[5] 페이로드 키 집합")
    full = hist[10][2]
    stale = hist[44][2]                  # 마지막 프레임(t33) 후 0.73 s
    print("   검출 있음: %s" % json.dumps({k: v for k, v in full.items() if k != "items"}))
    print("   전부 끊김: %s" % json.dumps(stale))
    print("   item 예:  %s" % json.dumps(full["items"][0]))
    check(set(full) == set(stale) == {"detected", "n", "frame_id", "items",
                                      "image_stamp", "age_ms"},
          "검출 유무와 무관하게 키 동일 (+ stamp 는 HeartbeatPublisher)")
    check(stale["image_stamp"] is None and stale["age_ms"] is None and not stale["detected"],
          "소스가 전부 끊기면 image_stamp/age_ms = null")
    item_keys = {"id", "label", "source", "conf", "x", "y", "z", "width",
                 "x_err", "votes", "stable", "box"}
    check(all(set(i) == item_keys for _, _, p, _ in hist for i in p["items"]),
          "item 키 고정, side/sigma 없음")
    check(all(p["frame_id"] == "base_link" for _, _, p, _ in hist), "frame_id=base_link")

    # ── 6. 자가검증 민감도 (정보) ─────────────────────────────────────
    print("\n[6] 높이 자가검증이 pitch 오차를 몇 도부터 잡는가 (5 m 콘, 경고 기준 20%)")
    print("   실제 pitch 오차   x 오차     z       경고")
    for dp in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        Gt = from_params(lambda k, d, dp=dp: (d + dp) if k.endswith("cam_pitch_deg") else d)
        g = P.project(*cone_box(Gt, 5.0, 0.0))
        warn = abs(g["z"] - CONE_H) > 0.2 * CONE_H
        print("     %+.1f°         %+.2f m   %.3f   %s" % (dp, g["x"] - 5.0, g["z"],
                                                          "예" if warn else "아니오"))
    print("\n   검출기 세로 지터 ±2 px 일 때 z 흔들림 (거리별)")
    for X in (3.0, 5.0, 7.0, 10.0):
        x1, y1, x2, y2 = cone_box(G, X, 0.0)
        zs = [P.project(x1, y1 + a, x2, y2 + b)["z"] for a in (-2, 2) for b in (-2, 2)]
        print("     %4.1f m  박스 높이 %5.1f px  z %.3f ~ %.3f (±%.0f%%)"
              % (X, y2 - y1, min(zs), max(zs), 100 * (max(zs) - min(zs)) / 2 / CONE_H))

    print("\n" + ("전부 통과" if not FAILS else "실패 %d건: %s" % (len(FAILS), FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
