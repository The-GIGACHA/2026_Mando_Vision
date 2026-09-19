#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sign_car_view.py — 신호제어차량 판정을 눈으로 확인합니다 (catkin 빌드 불필요).

  배치 위치:  mando_vision_2026/tools/sign_car_view.py

  bag(또는 영상)을 직접 읽어 obstacle.pt + utils/sign_car.py 를
  차량 노드와 같은 조건(15 Hz, imgsz 960, conf 0.20)으로 돌리고 화면에 그립니다.

화면
  위 띠     NEW = 현재 sign_car.py 판정 (HELD 면 점멸 유지 중, 뒤 숫자는 경과 초)
            OLD = 예전 방식(패널이 차체 박스 안 + 유지 없음) — 비교용
  노란 박스 traffic_car        초록/빨강 박스  인정된 ↓ / X 패널
  회색 박스 버려진 패널 검출   하늘색 점선     패널 '제자리' (좌 패널 / 중앙 패널)
  아래 띠   최근 20초 타임라인 (초록=RIGHT 파랑=LEFT, 어두운 색=유지, 회색=없음)

키   space 일시정지   s 현재 화면 저장   q 종료

사용
  source /opt/ros/noetic/setup.bash
  python3 tools/sign_car_view.py /media/inji2/CREAM/h_20260919_150248.bag --start 205 --end 400
  python3 tools/sign_car_view.py some.bag --save out.mp4 --no-show     # 영상으로만 저장
  python3 tools/sign_car_view.py traffic_sign.mp4                      # 일반 영상도 됩니다
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

TOPIC = "/cam_front/color/image_raw/compressed"
COL = {"LEFT": (255, 140, 0), "RIGHT": (0, 220, 0)}
COL_HELD = {"LEFT": (140, 80, 0), "RIGHT": (0, 110, 0)}


def frames_of(path, a):
    """(초, 이미지) 를 냅니다. 초는 시작 기준."""
    if path.endswith(".bag"):
        import rosbag
        import rospy
        bag = rosbag.Bag(path)
        t0 = bag.get_start_time()
        end = rospy.Time(t0 + a.end) if a.end else None
        for _t, m, ts in bag.read_messages(topics=[a.topic],
                                           start_time=rospy.Time(t0 + a.start),
                                           end_time=end):
            im = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                yield ts.to_sec() - t0, im
        bag.close()
    else:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        i = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            sec = i / fps
            i += 1
            if sec < a.start:
                continue
            if a.end and sec > a.end:
                break
            yield sec, im
        cap.release()


def dashed_rect(img, p1, p2, col):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    for x in range(x1, x2, 12):
        cv2.line(img, (x, y1), (min(x + 6, x2), y1), col, 1)
        cv2.line(img, (x, y2), (min(x + 6, x2), y2), col, 1)
    for y in range(y1, y2, 12):
        cv2.line(img, (x1, y), (x1, min(y + 6, y2)), col, 1)
        cv2.line(img, (x2, y), (x2, min(y + 6, y2)), col, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help=".bag 또는 영상 파일")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--model", default=os.path.join(here, "..", "weights", "obstacle.pt"))
    ap.add_argument("--utils", default=os.path.join(here, "..", "utils"))
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--hz", type=float, default=15.0, help="판정 주기 (차량 노드 = 15)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--speed", type=float, default=1.0, help="재생 배속 (0 = 최대한 빨리)")
    ap.add_argument("--save", default=None, help="mp4 로 저장")
    ap.add_argument("--no-show", dest="show", action="store_false")
    a = ap.parse_args()

    sys.path.insert(0, a.utils)
    from sign_car import SignCarParams, SignCarVoter, decide, PANELS, HOST, _in_zone

    p_new = SignCarParams()
    p_old = SignCarParams()
    p_old.gate_mode, p_old.hold_s = "inside", 0.0
    v_new, v_old = SignCarVoter(p_new), SignCarVoter(p_old)

    net = YOLO(a.model)
    names = net.names
    hist = deque(maxlen=int(20 * a.hz))
    writer = None
    next_t = -1.0
    n = n_cmd = n_held = 0
    wall0 = sec0 = None
    paused = False

    for sec, im in frames_of(a.src, a):
        if sec < next_t:
            continue
        next_t = sec + 1.0 / a.hz - 1e-3

        r = net.predict(im, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        dets = [dict(name=names[int(b.cls)], conf=float(b.conf),
                     box=tuple(b.xyxy[0].tolist())) for b in r.boxes]
        o_new = decide(dets, p_new)
        cmd, votes = v_new.update(o_new, now=sec)
        cmd_old, _ = v_old.update(decide(dets, p_old), now=sec)
        n += 1
        n_cmd += cmd is not None
        n_held += bool(v_new.held)
        hist.append((cmd, v_new.held))

        # ── 그리기 ───────────────────────────────────────────────
        vis = im.copy()
        H, W = vis.shape[:2]
        hosts = [d for d in dets if d["name"] == HOST]
        for h in hosts:
            x1, y1, x2, y2 = h["box"]
            hw, hh, cx = x2 - x1, y2 - y1, (x1 + x2) / 2.0
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(vis, "traffic_car %.2f" % h["conf"], (int(x1), int(y2) + 16),
                        0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            for zx, lab in ((p_new.zone_left, "L panel"), (p_new.zone_right, "R(center) panel")):
                q1 = (int(cx + zx[0] * hw), int(y1 - p_new.zone_up[1] * hh))
                q2 = (int(cx + zx[1] * hw), int(y1 - p_new.zone_up[0] * hh))
                dashed_rect(vis, q1, q2, (255, 200, 80))
                cv2.putText(vis, lab, (q1[0], max(70, q1[1] - 4)), 0, 0.4, (255, 200, 80), 1, cv2.LINE_AA)
        for d in dets:
            if d["name"] not in PANELS:
                continue
            ok = (d["conf"] >= (p_new.conf_go if d["name"].endswith("_go") else p_new.conf_x)
                  and any(_in_zone(d["name"], d["box"], h["box"], p_new) for h in hosts))
            col = ((0, 255, 0) if d["name"].endswith("_go") else (0, 0, 255)) if ok else (150, 150, 150)
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, "%s %.2f%s" % (d["name"], d["conf"], "" if ok else " (drop)"),
                        (x1, max(70, y1 - 5)), 0, 0.5, col, 2, cv2.LINE_AA)

        cv2.rectangle(vis, (0, 0), (W, 60), (0, 0, 0), -1)
        state = "HELD %.1fs" % v_new.held_s if v_new.held else ("SEEN %d/%d" % (votes, p_new.vote_n) if cmd else "")
        cv2.putText(vis, "NEW: %-5s %s" % (cmd or "---", state), (10, 26), 0, 0.9,
                    (COL_HELD if v_new.held else COL).get(cmd, (130, 130, 130)), 2, cv2.LINE_AA)
        cv2.putText(vis, "OLD: %s" % (cmd_old or "---"), (470, 26), 0, 0.8,
                    COL.get(cmd_old, (130, 130, 130)), 2, cv2.LINE_AA)
        cv2.putText(vis, "t=%d:%04.1f" % (int(sec) // 60, sec % 60), (W - 170, 26), 0, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        sub = "basis=%s conf=%s both=%s conflict=%s  %s  drop=%s" % (
            o_new["basis"], o_new["confidence"], o_new["both_seen"], o_new["conflict"],
            " ".join("%s=%.2f" % (k, v) for k, v in o_new["panels"].items() if v),
            o_new["dropped"] or "")
        cv2.putText(vis, sub, (10, 50), 0, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

        cv2.rectangle(vis, (0, H - 16), (W, H), (40, 40, 40), -1)
        bw = W / float(hist.maxlen)
        for k, (c, held) in enumerate(hist):
            if c:
                cv2.rectangle(vis, (int(k * bw), H - 14), (int((k + 1) * bw), H - 2),
                              (COL_HELD if held else COL)[c], -1)

        if a.save:
            if writer is None:
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), a.hz, (W, H))
            writer.write(vis)
        if a.show:
            cv2.imshow("sign_car_view", vis)
            if wall0 is None:
                wall0, sec0 = time.time(), sec
            wait = 1
            if a.speed > 0:
                wait = max(1, int(1000 * ((sec - sec0) / a.speed - (time.time() - wall0))))
            while True:
                k = cv2.waitKey(wait if not paused else 50) & 0xFF
                if k == ord(" "):
                    paused = not paused
                    wall0, sec0 = time.time(), sec
                elif k == ord("s"):
                    fn = "sign_car_%07.1fs.jpg" % sec
                    cv2.imwrite(fn, vis)
                    print("저장:", fn)
                elif k == ord("q"):
                    n = -abs(n)
                if not paused or n < 0:
                    break
            if n < 0:
                break

    if writer is not None:
        writer.release()
    n = abs(n)
    if n:
        print("프레임 %d  지시 나간 비율 %.0f%% (그중 유지 %.0f%%)"
              % (n, 100.0 * n_cmd / n, 100.0 * n_held / max(1, n_cmd)))
    if a.save:
        print("-> %s" % a.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
