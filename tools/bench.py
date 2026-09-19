#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench.py — .pt 와 .engine 의 속도를 재고, 검출이 같은지 대조합니다.

  배치 위치:  mando_vision_2026/tools/bench.py
  ★ 반드시 '차량 노트북'에서 돌리세요. 랩실 PC 수치는 의미가 없습니다.

────────────────────────────────────────────────────────────────────────
왜 대조까지 하는가

TensorRT 변환은 조용히 실패합니다. 에러 없이 엔진이 만들어지는데
검출이 달라지는 경우가 있습니다. 흔한 원인:

  · export 의 imgsz 가 학습·런타임과 다름
  · FP16 로 내리면서 작은 객체의 점수가 임계 아래로 떨어짐
  · NMS 설정 차이

속도만 재고 넘어가면 대회 당일에 "노트북에서는 잘 됐는데" 가 됩니다.
그래서 같은 프레임에 두 모델을 돌려 클래스별 검출 수를 비교합니다.

────────────────────────────────────────────────────────────────────────
사용

  # .pt 와 .engine 을 같이 재고 대조
  python3 tools/bench.py \\
      --model weights/obstacle.pt weights/obstacle.engine \\
      --src videos/driving.mp4 --imgsz 960 --n 200

  # 신호등
  python3 tools/bench.py \\
      --model weights/traffic_light.pt weights/traffic_light.engine \\
      --src videos/driving.mp4 --imgsz 640 --n 200

  # 영상 없이 (합성 프레임으로 속도만)
  python3 tools/bench.py --model weights/obstacle.engine --imgsz 960 --n 100
"""

import argparse
import os
import statistics
import sys
import time
from collections import Counter

import numpy as np
import cv2


def load_frames(src, n, size):
    """대조는 '같은 프레임'으로 해야 의미가 있어 미리 메모리에 올립니다."""
    if not src:
        rng = np.random.default_rng(0)
        return [rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
                for _ in range(min(n, 30))]
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit("영상을 열 수 없습니다: %s" % src)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.resize(f, size, interpolation=cv2.INTER_LINEAR))
    cap.release()
    if not out:
        sys.exit("프레임을 읽지 못했습니다")
    return out


def run(path, frames, imgsz, conf, warmup):
    from ultralytics import YOLO
    import torch

    is_engine = path.endswith(".engine")
    m = YOLO(path, task="detect")
    dev = 0 if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and is_engine:
        sys.exit("엔진은 GPU 가 있어야 돕니다.")

    for f in frames[:warmup]:
        m.predict(f, imgsz=imgsz, conf=conf, device=dev, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    counts = Counter()
    per_frame = []
    for f in frames:
        t0 = time.perf_counter()
        r = m.predict(f, imgsz=imgsz, conf=conf, device=dev, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

        c = Counter()
        if r.boxes is not None:
            for b in r.boxes:
                c[int(b.cls[0])] += 1
        counts.update(c)
        per_frame.append(c)

    names = {int(k): str(v) for k, v in dict(m.names).items()}
    return dict(path=path, times=times, counts=counts,
                per_frame=per_frame, names=names)


def stats(t):
    t = sorted(t)
    return dict(mean=statistics.fmean(t), med=t[len(t) // 2],
                p95=t[min(len(t) - 1, int(len(t) * 0.95))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True,
                    help="첫 번째가 기준(보통 .pt), 나머지를 대조")
    ap.add_argument("--src", default=None, help="영상 (없으면 합성 프레임)")
    ap.add_argument("--imgsz", type=int, required=True,
                    help="★ 학습·export 와 같은 값")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--size", nargs=2, type=int, default=[1280, 720],
                    help="입력 프레임 크기 (실제 카메라와 같게)")
    a = ap.parse_args()

    for p in a.model:
        if not os.path.isfile(p):
            sys.exit("파일이 없습니다: %s" % p)

    frames = load_frames(a.src, a.n, tuple(a.size))
    print("=" * 70)
    print("프레임 %d장  입력 %dx%d  imgsz=%d  conf=%.2f"
          % (len(frames), a.size[0], a.size[1], a.imgsz, a.conf))
    print("=" * 70)

    res = []
    for p in a.model:
        print("\n[%s] 측정 중..." % os.path.basename(p))
        res.append(run(p, frames, a.imgsz, a.conf, a.warmup))

    # ── 속도 ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("속도")
    print("=" * 70)
    print("   %-28s %8s %8s %8s %8s" % ("모델", "평균ms", "중앙ms", "p95ms", "FPS"))
    for r in res:
        s = stats(r["times"])
        print("   %-28s %8.1f %8.1f %8.1f %8.1f"
              % (os.path.basename(r["path"]), s["mean"], s["med"], s["p95"],
                 1000.0 / max(s["mean"], 1e-6)))
    if len(res) > 1:
        b, e = stats(res[0]["times"])["mean"], stats(res[-1]["times"])["mean"]
        print("\n   가속 %.2f배  (%.1f ms -> %.1f ms)" % (b / max(e, 1e-6), b, e))

    print("\n   ★ p95 를 보세요. 평균이 아니라 이 값이 최악의 주기를 정합니다.")
    print("     주기 배분은 p95 기준으로 잡아야 실제로 안 밀립니다.")

    # ── 검출 대조 ────────────────────────────────────────────────
    if len(res) > 1:
        print("\n" + "=" * 70)
        print("검출 대조  (기준: %s)" % os.path.basename(res[0]["path"]))
        print("=" * 70)
        base = res[0]
        names = base["names"]
        allc = set(base["counts"]) | set().union(*[set(r["counts"]) for r in res[1:]])
        print("   %-14s %10s %10s %10s" % ("클래스", "기준", "대조", "차이"))
        worst = 0.0
        for c in sorted(allc):
            nb = base["counts"].get(c, 0)
            ne = res[-1]["counts"].get(c, 0)
            d = (ne - nb) / max(nb, 1) * 100.0
            worst = max(worst, abs(d))
            flag = "  ←★" if abs(d) > 15 else ""
            print("   %-14s %10d %10d %9.1f%%%s"
                  % (names.get(c, str(c)), nb, ne, d, flag))

        # 프레임 단위 일치율
        agree = sum(1 for a_, b_ in zip(base["per_frame"], res[-1]["per_frame"])
                    if a_ == b_)
        print("\n   프레임 단위 완전일치 %d/%d (%.0f%%)"
              % (agree, len(frames), 100.0 * agree / len(frames)))

        print("\n   판정")
        if worst <= 5:
            print("   정상 — 엔진을 그대로 쓰세요.")
        elif worst <= 15:
            print("   경미한 차이 — FP16 양자화로 흔한 수준입니다.")
            print("   다만 검출이 '줄어든' 클래스가 있으면 conf 를 0.05 낮춰 보세요.")
        else:
            print("   ★ 차이가 큽니다. 엔진을 그대로 쓰지 마세요.")
            print("     확인할 것:")
            print("      1) export 의 imgsz 가 %d 였는가" % a.imgsz)
            print("      2) 작은 클래스가 줄었다면 FP16 문제 — half=False 로 재변환")
            print("      3) 학습 때와 다른 ultralytics 버전으로 export 하지 않았는가")
    return 0


if __name__ == "__main__":
    sys.exit(main())
