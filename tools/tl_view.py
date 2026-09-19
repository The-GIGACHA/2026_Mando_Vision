#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tl_view.py — 신호등 판정을 눈으로 확인합니다. 여러 모델을 한 화면에서 비교합니다.

  배치 위치:  mando_vision_2026/tools/tl_view.py

  bag(또는 영상)을 직접 읽어 traffic_light_detector.py 와 같은 규칙으로 돌립니다.
      10 Hz · conf 0.5 · 프레임당 가장 신뢰도 높은 박스 1개 · 5프레임 중 3표로 확정

화면
  위 띠      모델마다 한 줄:  확정 라벨(표수)  |  이번 프레임의 raw 검출
  굵은 박스  그 모델이 이번 프레임에 고른 박스 (모델별 테두리 색, 글자는 라벨 색)
  가는 회색  conf 0.25~임계 사이라 버려진 검출 — '아깝게 놓친 것'과 '잠재 오검출'이 보입니다
  자홍 박스  --roi 영역
  아래 띠    모델마다 최근 20초 확정 라벨 타임라인
             (빨강 red / 노랑 yellow / 초록 green / 청록 green_left / 파랑 left / 회색 없음)

키   space 일시정지   s 현재 화면 저장   q 종료

사용
  source /opt/ros/noetic/setup.bash
  # 역광 bag — classes.yaml 에 적힌 가중치 하나로 (기본)
  python3 tools/tl_view.py /media/inji2/CREAM/traffic_light_20260919_135705.bag
  # 모델 지정 (몇 개든)
  python3 tools/tl_view.py some.bag --models weights/traffic_light_20250918.pt weights/traffic_light_ft640_v2_20260920.pt
  # ROI 크롭 효과 보기 (원본 좌표 x1 y1 x2 y2)
  python3 tools/tl_view.py some.bag --roi 320 80 960 440
  # 창 없이 mp4 로만
  python3 tools/tl_view.py some.bag --start 0 --end 60 --save tl_check.mp4 --no-show
"""

import argparse
import os
import sys
import time
from collections import Counter, deque

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sign_car_view import frames_of, TOPIC   # noqa: E402  (bag/영상 읽기는 같은 코드)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LABEL_COL = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 220, 0),
             "green_left": (200, 220, 0), "left": (255, 140, 0)}
MODEL_COL = [(255, 255, 255), (255, 0, 255), (0, 165, 255), (255, 255, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help=".bag 또는 영상 파일")
    ap.add_argument("--models", nargs="+",
                    default=None, help="가중치 경로들 (기본: config/classes.yaml 의 traffic_light.file)")
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--hz", type=float, default=10.0, help="판정 주기 (차량 노드 = 10)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5, help="노드의 ~confidence")
    ap.add_argument("--vote-window", dest="vote_n", type=int, default=5)
    ap.add_argument("--vote-min", dest="vote_k", type=int, default=3)
    ap.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--speed", type=float, default=1.0, help="재생 배속 (0 = 최대한 빨리)")
    ap.add_argument("--save", default=None, help="mp4 로 저장")
    ap.add_argument("--no-show", dest="show", action="store_false")
    a = ap.parse_args()

    if not a.models:
        import yaml
        with open(os.path.join(ROOT, "config", "classes.yaml"), encoding="utf-8") as f:
            a.models = [os.path.join(ROOT, yaml.safe_load(f)["traffic_light"]["file"])]
    nets = []
    for k, p in enumerate(a.models):
        if not os.path.isfile(p):
            sys.exit("모델 파일이 없습니다: %s" % p)
        tag = os.path.splitext(os.path.basename(p))[0].replace("traffic_light_", "")
        nets.append(dict(tag=tag[:18], net=YOLO(p), col=MODEL_COL[k % len(MODEL_COL)],
                         votes=deque(maxlen=a.vote_n), hist=deque(maxlen=int(20 * a.hz)),
                         stat=Counter(), flips=0, last=None))

    LOW = min(0.25, a.conf)
    banner = 8 + 24 * len(nets)
    writer = None
    next_t = -1.0
    n = 0
    wall0 = sec0 = None
    paused = quit_ = False

    for sec, im in frames_of(a.src, a):
        if sec < next_t:
            continue
        next_t = sec + 1.0 / a.hz - 1e-3
        n += 1
        H, W = im.shape[:2]
        if a.roi:
            x1, y1 = max(0, min(a.roi[0], W - 1)), max(0, min(a.roi[1], H - 1))
            x2, y2 = max(x1 + 1, min(a.roi[2], W)), max(y1 + 1, min(a.roi[3], H))
        else:
            x1, y1, x2, y2 = 0, 0, W, H
        crop = im[y1:y2, x1:x2]

        vis = im.copy()
        if a.roi:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 1)
        lines = []
        for k, m in enumerate(nets):
            r = m["net"].predict(crop, imgsz=a.imgsz, conf=LOW, verbose=False)[0]
            dets = sorted(((float(b.conf), m["net"].names[int(b.cls)],
                            [v + o for v, o in zip(b.xyxy[0].tolist(), (x1, y1, x1, y1))])
                           for b in r.boxes), reverse=True)
            top = dets[0] if dets and dets[0][0] >= a.conf else None
            m["votes"].append(top[1] if top else None)
            lab, cnt = Counter(m["votes"]).most_common(1)[0]
            stable = lab if (lab is not None and cnt >= a.vote_k) else None
            m["hist"].append(stable)
            m["stat"][stable] += 1
            if stable and m["last"] and stable != m["last"]:
                m["flips"] += 1
            if stable:
                m["last"] = stable

            for c, name, bx in dets:
                q = [int(v) for v in bx]
                if top is not None and (c, name, bx) == top:
                    cv2.rectangle(vis, (q[0] - 2 * k, q[1] - 2 * k), (q[2] + 2 * k, q[3] + 2 * k), m["col"], 2)
                    cv2.putText(vis, "%s %.2f" % (name, c), (q[0], q[3] + 16 + 16 * k), 0, 0.5,
                                LABEL_COL.get(name, (200, 200, 200)), 2, cv2.LINE_AA)
                elif c < a.conf:
                    cv2.rectangle(vis, (q[0], q[1]), (q[2], q[3]), (160, 160, 160), 1)
                    cv2.putText(vis, "%s %.2f" % (name, c), (q[0], max(banner + 12, q[1] - 4)), 0, 0.4,
                                (160, 160, 160), 1, cv2.LINE_AA)
            raw = ("%s %.2f w%d" % (top[1], top[0], top[2][2] - top[2][0])) if top else (
                "(%s %.2f < %.2f)" % (dets[0][1], dets[0][0], a.conf) if dets else "-")
            lines.append((m, stable, cnt, raw))

        cv2.rectangle(vis, (0, 0), (W, banner), (0, 0, 0), -1)
        for k, (m, stable, cnt, raw) in enumerate(lines):
            y = 22 + 24 * k
            cv2.rectangle(vis, (6, y - 13), (18, y - 1), m["col"], -1)
            cv2.putText(vis, m["tag"], (26, y), 0, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(vis, "%-10s %d/%d" % (stable or "---", cnt if stable else 0, a.vote_n),
                        (250, y), 0, 0.7, LABEL_COL.get(stable, (130, 130, 130)), 2, cv2.LINE_AA)
            cv2.putText(vis, "raw: " + raw, (520, y), 0, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(vis, "t=%d:%04.1f" % (int(sec) // 60, sec % 60), (W - 150, 22), 0, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

        strip = 12
        for k, m in enumerate(nets):
            yb = H - strip * (len(nets) - k)
            cv2.rectangle(vis, (0, yb), (W, yb + strip), (40, 40, 40), -1)
            cv2.rectangle(vis, (0, yb + 2), (6, yb + strip - 2), m["col"], -1)
            bw = (W - 8) / float(m["hist"].maxlen)
            for i, s in enumerate(m["hist"]):
                if s:
                    cv2.rectangle(vis, (8 + int(i * bw), yb + 2), (8 + int((i + 1) * bw), yb + strip - 2),
                                  LABEL_COL.get(s, (200, 200, 200)), -1)

        if a.save:
            if writer is None:
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), a.hz, (W, H))
            writer.write(vis)
        if a.show:
            cv2.imshow("tl_view", vis)
            if wall0 is None:
                wall0, sec0 = time.time(), sec
            wait = 1
            if a.speed > 0:
                wait = max(1, int(1000 * ((sec - sec0) / a.speed - (time.time() - wall0))))
            while True:
                key = cv2.waitKey(wait if not paused else 50) & 0xFF
                if key == ord(" "):
                    paused = not paused
                    wall0, sec0 = time.time(), sec
                elif key == ord("s"):
                    fn = "tl_%07.1fs.jpg" % sec
                    cv2.imwrite(fn, vis)
                    print("저장:", fn)
                elif key == ord("q"):
                    quit_ = True
                if not paused or quit_:
                    break
            if quit_:
                break

    if writer is not None:
        writer.release()
    if n:
        print("\n프레임 %d (%.0f Hz)" % (n, a.hz))
        for m in nets:
            s = m["stat"]
            print("  %-20s 확정 비율 %3.0f%%  %s  라벨 바뀜 %d회"
                  % (m["tag"], 100.0 * (n - s[None]) / n,
                     {k: v for k, v in s.most_common() if k}, m["flips"]))
    if a.save:
        print("-> %s" % a.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
