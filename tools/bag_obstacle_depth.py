#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bag_obstacle_depth.py — 녹화된 bag 으로 '뎁스 vs 지면 투영' 을 ROS 없이 비교한다.

  python3 tools/bag_obstacle_depth.py BAG [--mission S_COURSE] [--labels T870 cone_yellow cone_blue]
                                          [--sheet out.jpg] [--every 5]

bag 에 있어야 하는 토픽
  /perception/obstacle                                         당시 ground_markers 출력 (bbox 원천)
  /cam_front/aligned_depth_to_color/image_raw/compressedDepth  정렬 뎁스
  /cam_front/color/image_raw/compressed                        --sheet 일 때만
  /global_mission                                              --mission 일 때만

박스는 녹화된 /perception/obstacle 의 box 를 그대로 쓰고, 그 image_stamp 와 같은 시각의
뎁스 프레임에 utils/obstacle_depth.py 를 돌린다. 모델을 다시 돌리지 않는다.
역광 bag 을 새로 따면 이 도구로 mismatch / few_valid 비율이 어떻게 변하는지 보면 된다.
"""
import argparse
import bisect
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils"))

import numpy as np                                              # noqa: E402
import rosbag                                                   # noqa: E402

from ground import from_params                                  # noqa: E402
from obstacle_depth import DepthProjector, decode_compressed_depth  # noqa: E402

DEPTH = "/cam_front/aligned_depth_to_color/image_raw/compressedDepth"
COLOR = "/cam_front/color/image_raw/compressed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--mission", default="S_COURSE", help="빈 문자열이면 전 구간")
    ap.add_argument("--labels", nargs="+", default=["T870", "cone_yellow", "cone_blue"])
    ap.add_argument("--sheet", default="", help="판정 예시를 모은 jpg 경로")
    ap.add_argument("--every", type=int, default=5, help="시트에 N 번째 항목마다 한 장")
    ap.add_argument("--pitch", type=float, default=None, help="ground.py cam_pitch_deg 덮어쓰기 (위를 보면 -)")
    ap.add_argument("--height", type=float, default=None, help="ground.py cam_height 덮어쓰기 [m]")
    a = ap.parse_args()

    bag = rosbag.Bag(a.bag)
    t0 = bag.get_start_time()

    # 1) 뎁스 프레임 위치만 먼저 모은다 (영상은 필요할 때 디코드)
    d_stamps, d_msgs = [], []
    for _t, msg, _ in bag.read_messages(topics=[DEPTH]):
        d_stamps.append(msg.header.stamp.to_sec())
        d_msgs.append(msg)
    if not d_msgs:
        print("뎁스 토픽 없음: %s" % DEPTH)
        return 1
    order = np.argsort(d_stamps)
    d_stamps = [d_stamps[i] for i in order]
    d_msgs = [d_msgs[i] for i in order]

    over = {"~ground/cam_pitch_deg": a.pitch, "~ground/cam_height": a.height}
    G = from_params(lambda k, d: d if over.get(k) is None else over[k])
    print("지면 투영: h=%.3f m  pitch=%+.2f°" % (G.h, np.degrees(G.th)))
    DP = DepthProjector(G)
    mission = None
    rows = []
    for topic, msg, t in bag.read_messages(topics=["/global_mission", "/perception/obstacle"]):
        if topic == "/global_mission":
            mission = msg.data
            continue
        if a.mission and mission != a.mission:
            continue
        d = json.loads(msg.data)
        st = d.get("image_stamp")
        items = [it for it in d.get("items") or [] if it.get("label") in a.labels]
        if st is None or not items:
            continue
        i = bisect.bisect_left(d_stamps, st)
        best = min((j for j in (i - 1, i) if 0 <= j < len(d_stamps)),
                   key=lambda j: abs(d_stamps[j] - st))
        dt = abs(d_stamps[best] - st)
        depth = None
        if dt <= 0.05:
            raw = decode_compressed_depth(d_msgs[best].data)
            depth = None if raw is None else raw.astype(np.float32) * 0.001
        for it in items:
            r = DP.measure(it["box"], depth)
            rows.append((t.to_sec() - t0, st, it["label"], r))

    if not rows:
        print("비교할 항목 없음 (mission=%r, labels=%s)" % (a.mission, a.labels))
        return 1

    cnt = collections.Counter((r["used"], r["reason"]) for _, _, _, r in rows)
    print("항목 %d개 (mission=%s)" % (len(rows), a.mission or "전체"))
    for (used, reason), n in cnt.most_common():
        print("  %-6s %-10s %4d  (%.0f%%)" % (used, reason, n, 100.0 * n / len(rows)))
    both = [(r["ground"]["x"], r["depth"]["x"], r["ground"]["y"], r["depth"]["y"])
            for _, _, _, r in rows if r["depth"] and "x" in r["depth"] and not r["cut"]]
    if both:
        g = np.array(both)
        dx = g[:, 1] - g[:, 0]
        dy = g[:, 3] - g[:, 2]
        print("뎁스 - 지면 (밑동 안 잘린 %d개)" % len(g))
        print("  x: 중앙값 %+.2f m  |x| p90 %.2f m  최대 %.2f m" %
              (np.median(dx), np.percentile(np.abs(dx), 90), np.abs(dx).max()))
        print("  y: 중앙값 %+.3f m  |y| p90 %.3f m  (좌우 부호 불일치 %d개)" %
              (np.median(dy), np.percentile(np.abs(dy), 90),
               int(np.sum(np.sign(g[:, 2]) != np.sign(g[:, 3])))))
        for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 99)):
            m = (g[:, 0] >= lo) & (g[:, 0] < hi)
            if m.any():
                print("  지면 %g~%g m: %3d개  x차 중앙값 %+.2f  |x차| p90 %.2f" %
                      (lo, hi, m.sum(), np.median(dx[m]), np.percentile(np.abs(dx[m]), 90)))
    if a.sheet:
        write_sheet(bag, rows[::max(1, a.every)], a.sheet)
    return 0


def write_sheet(bag, rows, path, max_tiles=24):
    import cv2
    rows = rows[:max_tiles]
    want = sorted(set(st for _, st, _, _ in rows))
    imgs = {}
    for _t, msg, _ in bag.read_messages(topics=[COLOR]):
        s = msg.header.stamp.to_sec()
        i = bisect.bisect_left(want, s - 1e-3)
        if i < len(want) and abs(want[i] - s) < 1e-3:
            imgs[want[i]] = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    tiles = []
    for tb, st, label, r in rows:
        im = imgs.get(st)
        if im is None:
            continue
        x1, y1, x2, y2 = r["box"]
        col = (0, 220, 0) if r["used"] == "depth" else (0, 140, 255)
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 3)
        gtxt = "g x=%.2f y=%+.2f" % (r["ground"]["x"], r["ground"]["y"]) if r["ground"] else "g -"
        dtxt = ("d x=%.2f y=%+.2f" % (r["depth"]["x"], r["depth"]["y"])
                if r["depth"] and "x" in r["depth"] else "d -")
        for k, s in enumerate(("%.1fs %s" % (tb, label), gtxt, dtxt,
                               "%s / %s" % (r["used"], r["reason"]))):
            cv2.putText(im, s, (20, 50 + 45 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6)
            cv2.putText(im, s, (20, 50 + 45 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 2)
        tiles.append(cv2.resize(im, (640, 360)))
    if not tiles:
        print("시트: 컬러 영상 없음")
        return
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)])
    cv2.imwrite(path, sheet)
    print("시트: %s (%d장)" % (path, len(tiles)))


if __name__ == "__main__":
    sys.exit(main())
