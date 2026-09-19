#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_frames.py — 영상에서 라벨링용 프레임을 뽑습니다.

  배치 위치:  mando_vision_2026/tools/extract_frames.py

────────────────────────────────────────────────────────────────────────
그냥 N프레임마다 저장하면 안 되는 이유

  ① 연속 프레임은 거의 같은 그림입니다
     0.5초 간격이라도 차가 서 있으면 100장이 사실상 1장입니다.
     라벨링 시간만 100배 쓰고 정보량은 그대로입니다.

  ② 흔들린 프레임은 라벨을 망칩니다
     모션블러 프레임에 박스를 치면 모델이 흐린 형태를 객체로 배웁니다.

  ③ 파일명에 출처가 없으면 나중에 train/val 을 못 나눕니다
     영상 프레임을 무작위 분할하면 같은 장면이 양쪽에 들어가
     val mAP 가 현장 성능과 무관해집니다.

────────────────────────────────────────────────────────────────────────
★ 중복 판정을 '평균 차이'로 하면 안 됩니다

썸네일 전체의 평균 밝기차를 쓰면, 화면의 5%만 차지하는 작은 물체가
움직여도 값이 거의 안 변합니다. 그런데 우리가 뽑으려는 게 바로 그
작은 물체(T870, 유아인형, 신호 패널)입니다. 평균으로 거르면 정작
필요한 변화를 '중복'으로 버립니다.

그래서 이 도구는 **픽셀 차이 상위 5%의 평균**을 씁니다.
화면 한구석에서만 생긴 변화도 제대로 잡힙니다.

임계값은 --min-diff auto 로 두면 영상마다 관측된 분포에서 정합니다.
영상이 밝든 어둡든, 정지 구간이 길든 짧든 알아서 맞춥니다.

────────────────────────────────────────────────────────────────────────
사용

  # ① 먼저 영상 훑어보기 — 어느 구간에 T870/인형이 나오는지 찾습니다
  python3 tools/extract_frames.py videos/driving.mp4 --probe 48

  #    -> probe_driving.jpg 에 시각이 찍힌 격자가 나옵니다

  # ② 그 구간만 뽑기
  python3 tools/extract_frames.py videos/driving.mp4 \\
      --clips 1:20-1:55,3:10-3:40 \\
      --out datasets/frames_t870 --every 0.4 --max 400

  # ③ 영상 여러 개 한 번에 (출처별로 파일명이 구분됩니다)
  python3 tools/extract_frames.py videos/*.mp4 --out datasets/frames_all

  # 너무 적게 나오면 임계를 직접 낮춥니다
  python3 tools/extract_frames.py videos/driving.mp4 --min-diff 3

결과
  <out>/images/driving_0083.4s_f002502.jpg
  <out>/manifest.csv        프레임별 시각·선명도·차이값
  <out>/preview.jpg         뽑힌 것들 격자

  images/ 폴더를 Roboflow 업로드에 그대로 드래그하면 됩니다.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import cv2

TW, TH = 128, 72          # 비교용 축소 크기
TOP_FRAC = 0.05           # 픽셀 차이 상위 5% 만 봅니다


def parse_time(s, fps):
    """'90' / '1:30' / '0:01:30' -> 초,  'f1234' -> 프레임번호를 초로"""
    s = s.strip()
    if s.lower().startswith("f"):
        return float(s[1:]) / fps
    try:
        vals = [float(p) for p in s.split(":")]
    except ValueError:
        raise SystemExit("시각 형식을 못 읽었습니다: %s" % s)
    sec = 0.0
    for v in vals:
        sec = sec * 60.0 + v
    return sec


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("영상을 열 수 없습니다: %s\n"
                         "  ffmpeg -i in.mov -c:v libx264 -pix_fmt yuv420p out.mp4\n"
                         "  로 변환 후 다시 시도하세요." % path)
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, n, w, h


def thumb(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (TW, TH), interpolation=cv2.INTER_AREA).astype(np.float32)


def local_diff(a, b):
    """픽셀 차이 상위 5% 의 평균.

    평균 차이와 달리, 화면 한구석의 작은 물체가 움직여도 값이 올라갑니다."""
    d = np.abs(a - b).ravel()
    k = max(1, int(d.size * TOP_FRAC))
    return float(np.partition(d, -k)[-k:].mean())


def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if g.shape[1] > 640:
        g = cv2.resize(g, (640, max(1, int(640 * g.shape[0] / g.shape[1]))))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def montage(frames, labels, cols=6, tile=(320, 180)):
    if not frames:
        return None
    rows = (len(frames) + cols - 1) // cols
    tw, th = tile
    sheet = np.full((rows * th, cols * tw, 3), 25, np.uint8)
    for i, (f, lab) in enumerate(zip(frames, labels)):
        r, c = divmod(i, cols)
        t = cv2.resize(f, (tw, th))
        cv2.rectangle(t, (0, th - 18), (tw, th), (0, 0, 0), -1)
        cv2.putText(t, lab, (4, th - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 255), 1, cv2.LINE_AA)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    return sheet


# ═══════════════════════════════════════════════════════════════════════
def probe(path, n_tiles, outdir):
    cap, fps, n, w, h = open_video(path)
    base = os.path.splitext(os.path.basename(path))[0]
    print("%s   %dx%d  %.1f fps  %d frames (%d:%02d)"
          % (base, w, h, fps, n, int(n / fps) // 60, int(n / fps) % 60))
    if h > w:
        print("   ! 세로 영상입니다. 추론은 가로 16:9 라 화각이 다릅니다.")
    if n <= 0:
        cap.release()
        return

    want = set(int(i) for i in np.linspace(0, max(n - 2, 0), n_tiles))
    frames, labels = [], []
    i = 0
    while cap.grab():
        if i in want:
            ok, f = cap.retrieve()
            if ok and f is not None:
                sec = i / fps
                frames.append(f)
                labels.append("%d:%04.1f  f%d" % (int(sec) // 60, sec % 60, i))
        i += 1
    cap.release()

    sheet = montage(frames, labels)
    if sheet is None:
        print("   프레임을 읽지 못했습니다")
        return
    p = os.path.join(outdir, "probe_%s.jpg" % base)
    cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("   -> %s" % p)
    print("   격자에서 대상이 보이는 구간의 시각을 읽어")
    print("   --clips 1:20-1:55 형태로 넘기세요.")


# ═══════════════════════════════════════════════════════════════════════
def scan(path, a):
    """1차 통과 — 샘플 프레임의 썸네일·선명도만 모읍니다 (원본은 안 들고 있음)."""
    cap, fps, n, w, h = open_video(path)
    base = os.path.splitext(os.path.basename(path))[0]

    if a.clips:
        ranges = []
        for c in a.clips.split(","):
            if "-" not in c:
                raise SystemExit("--clips 형식: 1:20-1:55,3:10-3:40")
            s, e = c.split("-", 1)
            ranges.append((parse_time(s, fps), parse_time(e, fps)))
    else:
        ranges = [(0.0, 1e9)]

    step = max(1, int(round(fps * a.every)))
    samples = []
    i = 0
    while cap.grab():
        sec = i / fps
        if (i % step) or not any(s <= sec <= e for s, e in ranges):
            i += 1
            continue
        ok, f = cap.retrieve()
        if ok and f is not None:
            samples.append(dict(idx=i, sec=sec, th=thumb(f),
                                sharp=sharpness(f)))
        i += 1
    cap.release()
    return base, fps, w, h, step, ranges, samples


def select(samples, a):
    """연속 차이 분포를 보고 임계를 정한 뒤 탐욕적으로 고릅니다."""
    if not samples:
        return [], 0.0, []
    diffs = [0.0]
    for k in range(1, len(samples)):
        diffs.append(local_diff(samples[k]["th"], samples[k - 1]["th"]))

    if str(a.min_diff).lower() == "auto":
        pool = sorted(d for d in diffs[1:] if d > 0)
        if pool:
            thr = max(1.5, pool[int(len(pool) * 0.35)])
        else:
            thr = 0.0
    else:
        thr = float(a.min_diff)

    keep = [0]
    last = samples[0]["th"]
    for k in range(1, len(samples)):
        if local_diff(samples[k]["th"], last) >= thr:
            keep.append(k)
            last = samples[k]["th"]
    return keep, thr, diffs


def write_selected(path, base, a, wanted, out_img, rows, preview):
    """2차 통과 — 고른 프레임만 실제로 디코드해 저장합니다."""
    cap, fps, _n, _w, _h = open_video(path)
    want = {s["idx"]: s for s in wanted}
    i = 0
    n_written = 0
    while cap.grab():
        if i in want:
            ok, f = cap.retrieve()
            if ok and f is not None:
                s = want[i]
                if a.resize and max(f.shape[:2]) > a.resize:
                    sc = a.resize / float(max(f.shape[:2]))
                    f = cv2.resize(f, None, fx=sc, fy=sc,
                                   interpolation=cv2.INTER_AREA)
                stem = "%s_%07.1fs_f%06d" % (a.prefix or base, s["sec"], i)
                cv2.imwrite(os.path.join(out_img, stem + ".jpg"), f,
                            [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                rows.append(dict(file=stem + ".jpg", source=base,
                                 sec=round(s["sec"], 2), frame=i,
                                 sharpness=round(s["sharp"], 1),
                                 diff=round(s.get("diff", 0.0), 2)))
                if len(preview) < a.preview:
                    preview.append((f, "%d:%04.1f" % (int(s["sec"]) // 60,
                                                      s["sec"] % 60)))
                n_written += 1
        i += 1
    cap.release()
    return n_written


# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="frames_out")
    ap.add_argument("--prefix", default=None,
                    help="파일명 접두 (기본: 영상 파일명). train/val 분할 기준")
    ap.add_argument("--every", type=float, default=0.4, help="샘플 간격 초")
    ap.add_argument("--clips", default=None, help="'1:20-1:55,3:10-3:40'")
    ap.add_argument("--min-diff", dest="min_diff", default="auto",
                    help="중복 임계. 'auto' 또는 숫자. 적게 나오면 숫자로 낮추세요")
    ap.add_argument("--drop-blurry", dest="drop_blurry", type=float, default=10.0,
                    help="선명도 하위 몇 퍼센트를 버릴지 (0 이면 안 버림)")
    ap.add_argument("--resize", type=int, default=1280, help="긴 변 최대 px (0=원본)")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--max", type=int, default=0, help="영상당 최대 장수 (0=제한없음)")
    ap.add_argument("--preview", type=int, default=30)
    ap.add_argument("--probe", type=int, default=0,
                    help="균등 샘플 격자만 만들고 끝냅니다")
    a = ap.parse_args()

    paths = []
    for v in a.videos:
        paths.extend(sorted(glob.glob(v)) or [v])
    for p in paths:
        if not os.path.isfile(p):
            sys.exit("파일이 없습니다: %s" % p)

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    if a.probe:
        for p in paths:
            probe(p, a.probe, out)
        return 0

    out_img = os.path.join(out, "images")
    os.makedirs(out_img, exist_ok=True)

    # ── 1차: 훑기 + 선택 ─────────────────────────────────────────
    plan = []
    for p in paths:
        base, fps, w, h, step, ranges, samples = scan(p, a)
        print("\n%s  %dx%d %.1f fps   구간 %s   %d프레임마다(%.2f초)"
              % (base, w, h, fps,
                 ("%s" % [("%.0f-%.0fs" % r) for r in ranges]) if a.clips else "전체",
                 step, a.every))
        if not samples:
            print("   샘플 없음 — --clips 구간이 영상 길이를 벗어났는지 확인하세요")
            continue

        keep, thr, diffs = select(samples, a)
        for k in keep:
            samples[k]["diff"] = diffs[k]
        chosen = [samples[k] for k in keep]
        if a.max:
            chosen = chosen[:a.max]

        d = sorted(d for d in diffs[1:])
        if d:
            print("   프레임 간 변화량:  10%% %.1f   중앙 %.1f   90%% %.1f"
                  % (d[len(d) // 10], d[len(d) // 2], d[len(d) * 9 // 10]))
        print("   샘플 %d장 -> 중복 제외 후 %d장  (임계 %.1f%s)"
              % (len(samples), len(chosen), thr,
                 ", auto" if str(a.min_diff).lower() == "auto" else ""))
        if len(chosen) < max(5, len(samples) * 0.05):
            print("   ! 너무 적게 남았습니다. --min-diff 를 숫자로 낮춰보세요 (예: %.1f)"
                  % max(1.0, thr * 0.5))
        plan.append((p, base, chosen))

    if not plan:
        sys.exit("뽑을 프레임이 없습니다")

    # ── 흔들린 프레임 제거 (전체 기준) ───────────────────────────
    if a.drop_blurry > 0:
        allsh = sorted(s["sharp"] for _p, _b, ch in plan for s in ch)
        if len(allsh) > 10:
            # 두 기준 중 엄격한 쪽
            #   ① 하위 N%            — 항상 조금은 걸러냄
            #   ② 중앙값의 25% 미만  — 흔들린 구간이 길어도 확실히 걸러냄
            #      (백분위만 쓰면 영상의 20%가 흔들렸을 때 절반이 살아남습니다)
            p_thr = allsh[int(len(allsh) * a.drop_blurry / 100.0)]
            m_thr = 0.25 * allsh[len(allsh) // 2]
            thr = max(p_thr, m_thr)
            before = sum(len(ch) for _p, _b, ch in plan)
            plan = [(p, b, [s for s in ch if s["sharp"] >= thr])
                    for p, b, ch in plan]
            after = sum(len(ch) for _p, _b, ch in plan)
            print("\n흔들린 프레임 제외: %d장 "
                  "(임계 %.1f = max(하위%.0f%% %.1f, 중앙값의 25%% %.1f))"
                  % (before - after, thr, a.drop_blurry, p_thr, m_thr))

    # ── 2차: 실제 저장 ───────────────────────────────────────────
    rows, preview = [], []
    for p, base, chosen in plan:
        if chosen:
            write_selected(p, base, a, chosen, out_img, rows, preview)

    with open(os.path.join(out, "manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "source", "sec", "frame",
                                           "sharpness", "diff"])
        wr.writeheader()
        wr.writerows(rows)

    if preview:
        sheet = montage([f for f, _ in preview], [l for _, l in preview])
        cv2.imwrite(os.path.join(out, "preview.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    print("\n" + "=" * 66)
    print("최종 %d장  ->  %s" % (len(rows), out_img))
    print("   목록: %s/manifest.csv" % out)
    if preview:
        print("   미리보기: %s/preview.jpg" % out)
    print("\n다음")
    print("   1. preview.jpg 로 대상이 실제로 담겼는지 확인")
    print("   2. images/ 를 Roboflow 업로드에 드래그")
    print("   3. ★ 이 프레임들은 같은 영상에서 나왔습니다.")
    print("      train/val 을 무작위로 나누면 같은 장면이 양쪽에 들어가")
    print("      val mAP 가 현장 성능과 무관해집니다.")
    print("      영상(파일명 접두) 단위로 통째로 나누세요.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
