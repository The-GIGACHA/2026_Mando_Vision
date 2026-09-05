#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cam_mosaic.py — 카메라 4대를 한 창에 2x2 로 띄웁니다.

현장에서 차를 움직이며 "어느 카메라가 언제 끊기는가"를 보기 위한 도구입니다.
rqt_image_view 창을 4개 띄우는 것보다 노트북 화면에서 훨씬 보기 편하고,
카메라별 실측 FPS 와 신호 끊김이 화면에 바로 표시됩니다.

  python3 cam_mosaic.py
  python3 cam_mosaic.py --size 480 360
  python3 cam_mosaic.py --compressed          # /compressed 토픽 구독 (대역폭 절약)
  python3 cam_mosaic.py --topics /a /b /c /d

  s : 현재 화면 스냅샷 저장
  f : 각 타일을 실제 해상도로 (창 크기 무시)
  q / ESC : 종료

테두리 색
  초록  정상 (10 fps 이상, 1초 내 수신)
  주황  프레임률 저하
  빨강  NO SIGNAL / STALE — 케이블·전원·노드를 의심하세요
"""

import argparse
import threading
import time

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

# (표시이름, 토픽) — 2x2 배치 순서
DEFAULT = [
    ("front  D455",     "/cam_front/color/image_raw"),
    ("left   BRIO",     "/cam_left/image_raw"),
    ("right  BRIO",     "/cam_right/image_raw"),
    ("stopline C920",   "/cam_stopline/image_raw"),
]

GREEN, ORANGE, RED, GREY = (0, 220, 0), (0, 165, 255), (0, 0, 255), (110, 110, 110)


class Tile(object):
    """토픽 하나의 최신 프레임과 수신 통계를 들고 있습니다."""

    def __init__(self, name, topic):
        self.name = name
        self.topic = topic
        self.img = None
        self.t = 0.0
        self.times = []
        self.n_total = 0
        self.lock = threading.Lock()
        self.bridge = CvBridge()

        if topic.endswith("/compressed"):
            rospy.Subscriber(topic, CompressedImage, self._cb_compressed,
                             queue_size=1, buff_size=2 ** 24)
        else:
            rospy.Subscriber(topic, Image, self._cb_raw,
                             queue_size=1, buff_size=2 ** 24)

    def _store(self, img):
        if img is None or img.size == 0:
            return
        now = time.time()
        with self.lock:
            self.img = img
            self.t = now
            self.n_total += 1
            self.times.append(now)
            if len(self.times) > 40:
                self.times.pop(0)

    def _cb_raw(self, msg):
        try:
            self._store(self.bridge.imgmsg_to_cv2(msg, "bgr8"))
        except Exception as e:
            rospy.logwarn_throttle(10.0, "[%s] %s", self.name, e)

    def _cb_compressed(self, msg):
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            self._store(cv2.imdecode(arr, cv2.IMREAD_COLOR))
        except Exception as e:
            rospy.logwarn_throttle(10.0, "[%s] %s", self.name, e)

    def snapshot(self):
        with self.lock:
            img = None if self.img is None else self.img.copy()
            age = (time.time() - self.t) if self.t else 1e9
            fps = 0.0
            if len(self.times) > 1:
                span = self.times[-1] - self.times[0]
                if span > 1e-6:
                    fps = (len(self.times) - 1) / span
            n = self.n_total
        return img, age, fps, n


def draw(tile, w, h):
    img, age, fps, n = tile.snapshot()

    if img is None:
        cell = np.zeros((h, w, 3), np.uint8)
        color, status = RED, "NO SIGNAL"
    elif age > 1.0:
        cell = cv2.resize(img, (w, h))
        cell = (cell * 0.35).astype(np.uint8)          # 어둡게 = 오래된 화면
        color, status = RED, "STALE %.1fs" % age
    else:
        cell = cv2.resize(img, (w, h))
        if fps >= 10.0:
            color, status = GREEN, ""
        else:
            color, status = ORANGE, "LOW FPS"

    if status:
        (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(cell, status, ((w - tw) // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # 상단 정보 바
    cv2.rectangle(cell, (0, 0), (w, 26), (0, 0, 0), -1)
    label = "%s   %.1f fps   n=%d" % (tile.name, fps, n)
    cv2.putText(cell, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, color, 1, cv2.LINE_AA)
    if img is not None:
        res = "%dx%d" % (img.shape[1], img.shape[0])
        (tw, _), _ = cv2.getTextSize(res, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(cell, res, (w - tw - 8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, GREY, 1, cv2.LINE_AA)

    cv2.rectangle(cell, (0, 0), (w - 1, h - 1), color, 2)
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", nargs=2, type=int, default=[480, 360],
                    metavar=("W", "H"), help="타일 하나의 크기")
    ap.add_argument("--compressed", action="store_true",
                    help="각 토픽에 /compressed 를 붙여 구독")
    ap.add_argument("--topics", nargs="*", default=None,
                    help="토픽을 직접 지정 (최대 4개)")
    a = ap.parse_args()

    rospy.init_node("cam_mosaic", anonymous=True, disable_signals=True)

    if a.topics:
        specs = [(t.strip("/").split("/")[0], t) for t in a.topics[:4]]
    else:
        specs = list(DEFAULT)
    if a.compressed:
        specs = [(n, t if t.endswith("/compressed") else t + "/compressed")
                 for n, t in specs]

    tiles = [Tile(n, t) for n, t in specs]
    for t in tiles:
        rospy.loginfo("구독: %-34s (%s)", t.topic, t.name)

    W, H = a.size
    win = "cam_mosaic   [s]nap  [f]ull  [q]uit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W * 2, H * 2)

    full = False
    rate = rospy.Rate(15)
    try:
        while not rospy.is_shutdown():
            if full:
                sizes = [t.snapshot()[0] for t in tiles]
                sizes = [s.shape[:2] for s in sizes if s is not None]
                if sizes:
                    H2 = max(s[0] for s in sizes)
                    W2 = max(s[1] for s in sizes)
                else:
                    W2, H2 = W, H
            else:
                W2, H2 = W, H

            cells = [draw(t, W2, H2) for t in tiles]
            while len(cells) < 4:
                blank = np.zeros((H2, W2, 3), np.uint8)
                cv2.putText(blank, "-", (W2 // 2, H2 // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREY, 2)
                cells.append(blank)

            grid = np.vstack([np.hstack(cells[0:2]), np.hstack(cells[2:4])])
            cv2.imshow(win, grid)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('s'):
                fn = time.strftime("mosaic_%Y%m%d_%H%M%S.png")
                cv2.imwrite(fn, grid)
                rospy.loginfo("저장: %s", fn)
            elif k == ord('f'):
                full = not full
                rospy.loginfo("full-res %s", "ON" if full else "OFF")
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print()
        print("── 최종 수신 통계 ──")
        for t in tiles:
            _, age, fps, n = t.snapshot()
            print("  %-16s %6.1f fps   총 %d 프레임   %s"
                  % (t.name, fps, n, "OK" if n > 0 else "수신 없음"))


if __name__ == "__main__":
    main()
