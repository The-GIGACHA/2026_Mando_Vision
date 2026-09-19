#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bag_frames.py — rosbag 의 CompressedImage 토픽에서 라벨링용 프레임을 뽑습니다.

  배치 위치:  mando_vision_2026/tools/bag_frames.py

  extract_frames.py 의 bag 판입니다. 중복·흔들림 판정은 그쪽 함수를 그대로 씁니다.

  ★ 영상으로 변환하지 않고 bag 에서 바로 뽑는 이유
    bag 안의 프레임은 이미 JPEG 입니다. mp4 로 바꿨다가 다시 jpg 로 저장하면
    손실 압축을 두 번 더 거칩니다. 여기서는 녹화된 JPEG 바이트를 그대로 씁니다.
    = 차량의 추론 노드가 본 것과 같은 픽셀입니다.

사용
  source /opt/ros/noetic/setup.bash

  # 훑어보기 — 어느 구간에 대상이 나오는지 (시각은 bag 시작 기준 초)
  python3 tools/bag_frames.py /media/inji2/CREAM/h_20260919_150248.bag --probe 48

  # 뽑기
  python3 tools/bag_frames.py /media/inji2/CREAM/h_2026*.bag \\
      --out frames_bag --every 0.5 --max 150

  # 구간만
  python3 tools/bag_frames.py some.bag --clips 1:20-1:55 --out frames_bag

결과는 extract_frames.py 와 같습니다 (images/ manifest.csv preview.jpg).
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import cv2
import rosbag

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_frames import (thumb, sharpness, montage, select,   # noqa: E402
                            parse_time)

TOPIC = "/cam_front/color/image_raw/compressed"


def decode(msg):
    return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)


def is_jpeg(msg):
    return bytes(msg.data[:2]) == b"\xff\xd8"


def clip_ranges(a):
    if not a.clips:
        return [(0.0, 1e9)]
    out = []
    for c in a.clips.split(","):
        if "-" not in c:
            raise SystemExit("--clips 형식: 1:20-1:55,3:10-3:40")
        s, e = c.split("-", 1)
        out.append((parse_time(s, 30.0), parse_time(e, 30.0)))
    return out


def probe(path, a, outdir):
    base = os.path.splitext(os.path.basename(path))[0]
    bag = rosbag.Bag(path)
    n = bag.get_message_count(a.topic)
    t0 = bag.get_start_time()
    dur = bag.get_end_time() - t0
    print("%s   %d frames (%d:%02d)" % (base, n, int(dur) // 60, int(dur) % 60))
    if n <= 0:
        print("   토픽이 없습니다: %s" % a.topic)
        return
    want = set(int(i) for i in np.linspace(0, n - 1, a.probe))
    frames, labels = [], []
    for i, (_t, m, ts) in enumerate(bag.read_messages(topics=[a.topic])):
        if i in want:
            f = decode(m)
            if f is not None:
                sec = ts.to_sec() - t0
                if not frames:
                    print("   %dx%d" % (f.shape[1], f.shape[0]))
                frames.append(f)
                labels.append("%d:%04.1f" % (int(sec) // 60, sec % 60))
    bag.close()
    p = os.path.join(outdir, "probe_%s.jpg" % base)
    cv2.imwrite(p, montage(frames, labels), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("   -> %s" % p)


def scan(path, a, ranges):
    """한 번만 읽습니다 — 외장 디스크의 bag 은 두 번 읽기엔 너무 느립니다.
    샘플 프레임의 JPEG 바이트를 메모리에 들고 있다가 고른 것만 씁니다."""
    bag = rosbag.Bag(path)
    t0 = bag.get_start_time()
    samples = []
    next_t = 0.0
    size = None
    for i, (_t, m, ts) in enumerate(bag.read_messages(topics=[a.topic])):
        sec = ts.to_sec() - t0
        if sec < next_t or not any(s <= sec <= e for s, e in ranges):
            continue
        f = decode(m)
        if f is None:
            continue
        next_t = sec + a.every
        size = (f.shape[1], f.shape[0])
        samples.append(dict(idx=i, sec=sec, th=thumb(f), sharp=sharpness(f),
                            data=bytes(m.data) if is_jpeg(m) else None,
                            img=None if is_jpeg(m) else f))
    bag.close()
    return samples, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--out", default="frames_bag")
    ap.add_argument("--every", type=float, default=0.5, help="샘플 간격 초")
    ap.add_argument("--clips", default=None, help="'1:20-1:55,3:10-3:40' (bag 시작 기준)")
    ap.add_argument("--min-diff", dest="min_diff", default="auto")
    ap.add_argument("--drop-blurry", dest="drop_blurry", type=float, default=10.0,
                    help="bag 마다 선명도 하위 몇 퍼센트를 버릴지 (0 이면 안 버림)")
    ap.add_argument("--max", type=int, default=0,
                    help="bag 당 최대 장수. 넘으면 시간축에서 균등하게 솎습니다")
    ap.add_argument("--preview", type=int, default=30)
    ap.add_argument("--probe", type=int, default=0)
    a = ap.parse_args()

    paths = []
    for v in a.bags:
        paths.extend(sorted(glob.glob(v)) or [v])
    for p in paths:
        if not os.path.isfile(p):
            sys.exit("파일이 없습니다: %s" % p)

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    if a.probe:
        for p in paths:
            probe(p, a, out)
        return 0

    out_img = os.path.join(out, "images")
    os.makedirs(out_img, exist_ok=True)
    ranges = clip_ranges(a)
    rows, preview = [], []

    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        samples, size = scan(p, a, ranges)
        if not samples:
            print("\n%s   샘플 없음 (토픽 %s)" % (base, a.topic))
            continue
        keep, thr, diffs = select(samples, a)
        for k in keep:
            samples[k]["diff"] = diffs[k]
        chosen = [samples[k] for k in keep]
        n_dedup = len(chosen)

        if a.drop_blurry > 0 and len(chosen) > 10:
            sh = sorted(s["sharp"] for s in chosen)
            bthr = max(sh[int(len(sh) * a.drop_blurry / 100.0)], 0.25 * sh[len(sh) // 2])
            chosen = [s for s in chosen if s["sharp"] >= bthr]
        n_sharp = len(chosen)

        if a.max and len(chosen) > a.max:
            pick = np.linspace(0, len(chosen) - 1, a.max).astype(int)
            chosen = [chosen[i] for i in pick]

        print("\n%s  %dx%d   샘플 %d -> 중복 제외 %d (임계 %.1f) -> 흔들림 제외 %d -> 저장 %d"
              % (base, size[0], size[1], len(samples), n_dedup, thr, n_sharp, len(chosen)))

        for s in chosen:
            stem = "%s_%07.1fs_f%06d" % (base, s["sec"], s["idx"])
            fp = os.path.join(out_img, stem + ".jpg")
            if s["data"] is not None:
                with open(fp, "wb") as f:
                    f.write(s["data"])
            else:
                cv2.imwrite(fp, s["img"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append(dict(file=stem + ".jpg", source=base, sec=round(s["sec"], 2),
                             frame=s["idx"], sharpness=round(s["sharp"], 1),
                             diff=round(s.get("diff", 0.0), 2)))
        step = max(1, len(chosen) // max(1, a.preview // len(paths)))
        for s in chosen[::step]:
            if len(preview) < a.preview:
                img = s["img"] if s["img"] is not None else cv2.imdecode(
                    np.frombuffer(s["data"], np.uint8), cv2.IMREAD_COLOR)
                preview.append((img, "%s %d:%02d" % (base[:14], int(s["sec"]) // 60,
                                                     int(s["sec"]) % 60)))
        del samples, chosen

    if not rows:
        sys.exit("뽑을 프레임이 없습니다")

    with open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "source", "sec", "frame",
                                           "sharpness", "diff"])
        wr.writeheader()
        wr.writerows(rows)
    sheet = montage([f for f, _ in preview], [l for _, l in preview])
    cv2.imwrite(os.path.join(out, "preview.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    print("\n" + "=" * 66)
    print("최종 %d장  ->  %s" % (len(rows), out_img))
    print("   train/val 은 bag(파일명 접두) 단위로 통째로 나누세요.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
