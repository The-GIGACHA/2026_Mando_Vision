#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
copypaste_aug.py — 객체를 오려 다른 배경에 붙여 학습 데이터를 늘립니다.

  배치 위치:  mando_vision_2026/tools/copypaste_aug.py

────────────────────────────────────────────────────────────────────────
왜 이 방식인가

밝기·회전·크기·이동 변형은 Ultralytics 가 학습 중에 매 에폭 이미 합니다.
오프라인으로 미리 만들어 두면 같은 변형을 '고정'시키는 것이라 오히려
다양성이 줄어듭니다. 장수만 늘고 정보량은 그대로입니다.

40장짜리 데이터셋에 정말 없는 것은 **배경 다양성**입니다.
그건 변형으로 만들 수 없고, 객체를 다른 배경에 옮겨 붙여야 생깁니다.

────────────────────────────────────────────────────────────────────────
★ 반드시 눈으로 확인하세요

합성 이미지는 경계선·조명 불일치 같은 인공 흔적을 남기고,
모델은 객체가 아니라 그 흔적을 검출하도록 학습할 수 있습니다.
현장에는 그 흔적이 없으므로 곧바로 미검출이 됩니다.

그래서 이 도구는 --preview 로 대조표(montage)를 만듭니다.
**붙인 자리가 눈에 띄면 그 설정은 버리세요.** 사람이 알아보는 경계선은
모델은 훨씬 더 쉽게 알아봅니다.

────────────────────────────────────────────────────────────────────────
★ 배경 이미지 주의

배경으로 쓰는 사진에 라벨 없는 T870/유아인형이 들어 있으면,
모델은 그것을 '배경'으로 학습합니다. 미검출을 직접 가르치는 셈입니다.
--bg 폴더에는 해당 객체가 절대 없는 프레임만 넣으세요.

────────────────────────────────────────────────────────────────────────
사용

  # 원본 데이터셋에서 T870/kid 를 오려 코스 배경에 붙입니다
  python3 tools/copypaste_aug.py \\
      --src datasets/obstacle_ours \\
      --bg  datasets/course_bg \\
      --classes T870,kid \\
      --out datasets/aug_t870_kid \\
      --n 300 --preview 24

  # 경계가 티나면 seamless(포아송) 합성으로
  python3 tools/copypaste_aug.py ... --blend seamless

  생성된 폴더의 images/ 와 labels/ 를 Roboflow 에 올려 원본과 합치세요.
  ★ 원본을 대체하는 게 아니라 '추가'입니다.
  ★ 합성 이미지는 절대 valid 로 보내지 마세요. train 전용입니다.
"""

import argparse
import os
import random
import sys

import numpy as np
import cv2

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_TARGET = ["T870", "kid", "traffic_car",
                  "left_go", "left_X", "right_go", "right_X"]


def load_yaml(p):
    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml 이 필요합니다:  pip install pyyaml --break-system-packages")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def names_of(d):
    n = d.get("names")
    if isinstance(n, dict):
        return [n[k] for k in sorted(n, key=lambda z: int(z))]
    return list(n or [])


def find_pairs(root):
    """(이미지경로, 라벨경로) 목록. labels 형제가 있는 images 폴더를 모두 훑습니다."""
    out = []
    for dp, _dn, fns in os.walk(root):
        if os.path.basename(dp).lower() != "images":
            continue
        lab = os.path.join(os.path.dirname(dp), "labels")
        if not os.path.isdir(lab):
            continue
        for fn in sorted(fns):
            if not fn.lower().endswith(IMG_EXT):
                continue
            lp = os.path.join(lab, os.path.splitext(fn)[0] + ".txt")
            out.append((os.path.join(dp, fn), lp if os.path.isfile(lp) else None))
    return out


def read_label(p):
    rows = []
    if not p:
        return rows
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.split()
            if len(s) < 5:
                continue
            try:
                rows.append((int(float(s[0])), [float(v) for v in s[1:5]]))
            except ValueError:
                pass
    return rows


def list_images(folder):
    out = []
    for dp, _dn, fns in os.walk(folder):
        for fn in sorted(fns):
            if fn.lower().endswith(IMG_EXT):
                out.append(os.path.join(dp, fn))
    return out


# ═══════════════════════════════════════════════════════════════════════
def build_alpha(h, w, shape, feather):
    """붙일 조각의 알파 마스크.

    직사각형 그대로 붙이면 네 모서리에 원래 배경이 따라와 상자가 보입니다.
    타원 마스크는 그 모서리를 잘라내므로 훨씬 자연스럽습니다."""
    a = np.zeros((h, w), np.float32)
    if shape == "ellipse":
        cv2.ellipse(a, (w // 2, h // 2), (int(w * 0.48), int(h * 0.48)),
                    0, 0, 360, 1.0, -1)
    else:
        m = max(1, int(min(h, w) * 0.06))
        a[m:h - m, m:w - m] = 1.0
    k = max(3, int(min(h, w) * feather) | 1)
    a = cv2.GaussianBlur(a, (k, k), 0)
    return np.clip(a, 0.0, 1.0)


def match_brightness(crop, bg_patch, strength):
    """붙일 조각의 밝기를 배경에 맞춥니다 (과하면 객체가 뭉개지니 제한)."""
    if strength <= 0:
        return crop
    cm = float(crop.reshape(-1, 3).mean()) + 1e-6
    bm = float(bg_patch.reshape(-1, 3).mean())
    r = bm / cm
    r = 1.0 + (r - 1.0) * strength
    r = float(np.clip(r, 0.6, 1.6))
    return np.clip(crop.astype(np.float32) * r, 0, 255).astype(np.uint8)


def iou_free(box, boxes, thr=0.02):
    x1, y1, x2, y2 = box
    for b in boxes:
        ix1, iy1 = max(x1, b[0]), max(y1, b[1])
        ix2, iy2 = min(x2, b[2]), min(y2, b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        if iw * ih > thr * (x2 - x1) * (y2 - y1):
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본 데이터셋 (data.yaml 있는 곳)")
    ap.add_argument("--bg", default=None,
                    help="배경 이미지 폴더. 생략하면 src 안의 '대상 클래스가 없는' 이미지")
    ap.add_argument("--classes", required=True, help="오려낼 클래스 (쉼표)")
    ap.add_argument("--target", default=",".join(DEFAULT_TARGET),
                    help="출력 클래스 순서")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300, help="생성할 이미지 장수")
    ap.add_argument("--per-image", dest="per_image", default="1,3",
                    help="한 장에 붙일 객체 수 범위 (예: 1,3)")
    ap.add_argument("--y-range", dest="y_range", default="0.45,0.92",
                    help="객체 '발밑'이 놓일 세로 구간 (화면 비율)")
    ap.add_argument("--size-range", dest="size_range", default=None,
                    help="붙일 높이 px 범위. 생략하면 원본 분포에서 추정")
    ap.add_argument("--blend", default="feather",
                    choices=["feather", "seamless"])
    ap.add_argument("--mask", default="ellipse", choices=["ellipse", "rect"])
    ap.add_argument("--feather", type=float, default=0.10)
    ap.add_argument("--bright-match", dest="bright", type=float, default=0.6,
                    help="배경 밝기 정합 강도 0~1")
    ap.add_argument("--preview", type=int, default=24,
                    help="대조표에 담을 장수 (0 이면 안 만듦)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)

    src = os.path.abspath(a.src)
    data = load_yaml(os.path.join(src, "data.yaml"))
    src_names = names_of(data)
    want = [s.strip() for s in a.classes.split(",") if s.strip()]
    for w in want:
        if w not in src_names:
            sys.exit("클래스 '%s' 가 원본에 없습니다. 있는 것: %s" % (w, src_names))
    want_ids = {src_names.index(w): w for w in want}

    target = [s.strip() for s in a.target.split(",") if s.strip()]
    T = {n: i for i, n in enumerate(target)}
    for w in want:
        if w not in T:
            sys.exit("target 에 '%s' 가 없습니다" % w)

    pairs = find_pairs(src)
    if not pairs:
        sys.exit("images/ + labels/ 를 찾지 못했습니다: %s" % src)

    # ── 조각 수집 ────────────────────────────────────────────────
    crops = {w: [] for w in want}
    clean_bgs = []
    for ip, lp in pairs:
        rows = read_label(lp)
        has = [c for c, _ in rows if c in want_ids]
        if not has:
            clean_bgs.append(ip)
            continue
        img = cv2.imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        for c, b in rows:
            if c not in want_ids:
                continue
            bw, bh = b[2] * W, b[3] * H
            x1 = int(round((b[0] - b[2] / 2) * W))
            y1 = int(round((b[1] - b[3] / 2) * H))
            x2 = int(round(x1 + bw))
            y2 = int(round(y1 + bh))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crops[want_ids[c]].append(img[y1:y2, x1:x2].copy())

    print("=" * 68)
    print("오려낸 조각")
    for w in want:
        hs = [c.shape[0] for c in crops[w]]
        if hs:
            hs.sort()
            print("   %-8s %4d개   높이 중앙 %d px (최소 %d / 최대 %d)"
                  % (w, len(hs), hs[len(hs) // 2], hs[0], hs[-1]))
        else:
            print("   %-8s    0개  ★ 조각이 없습니다" % w)
    if not any(crops[w] for w in want):
        sys.exit("오려낼 객체가 없습니다")

    bgs = list_images(a.bg) if a.bg else clean_bgs
    if not bgs:
        sys.exit("배경 이미지가 없습니다. --bg 로 폴더를 주세요")
    print("\n배경 이미지 %d장  (%s)"
          % (len(bgs), a.bg if a.bg else "src 안의 대상 없는 이미지"))
    if not a.bg:
        print("   ! src 배경을 재활용하므로 배경 다양성은 크게 늘지 않습니다.")
        print("     코스 영상 프레임 폴더를 --bg 로 주는 편이 훨씬 낫습니다.")

    pmin, pmax = [int(v) for v in a.per_image.split(",")]
    ymin, ymax = [float(v) for v in a.y_range.split(",")]
    if a.size_range:
        smin, smax = [int(v) for v in a.size_range.split(",")]
    else:
        allh = sorted(c.shape[0] for w in want for c in crops[w])
        smin = max(16, int(allh[int(len(allh) * 0.15)] * 0.7))
        smax = max(smin + 8, int(allh[int(len(allh) * 0.85)] * 1.3))
        print("   붙일 높이 범위를 원본 분포에서 추정: %d~%d px" % (smin, smax))

    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit("출력 폴더가 비어있지 않습니다: %s" % out)
    os.makedirs(os.path.join(out, "images"))
    os.makedirs(os.path.join(out, "labels"))

    made = 0
    placed = {w: 0 for w in want}
    previews = []
    tries = 0
    while made < a.n and tries < a.n * 20:
        tries += 1
        bg = cv2.imread(random.choice(bgs))
        if bg is None:
            continue
        canvas = bg.copy()
        H, W = canvas.shape[:2]
        boxes = []
        rows = []

        for _ in range(random.randint(pmin, pmax)):
            w = random.choice([w for w in want if crops[w]])
            crop = random.choice(crops[w])
            ch, cw = crop.shape[:2]

            # 아래쪽에 놓을수록 크게 — 원근 규칙
            yb = random.uniform(ymin, ymax)
            t = (yb - ymin) / max(ymax - ymin, 1e-6)
            h_new = int(round((smin + (smax - smin) * t)
                              * random.uniform(0.85, 1.15)))
            h_new = max(16, min(h_new, int(H * 0.6)))
            w_new = max(8, int(round(h_new * cw / float(ch))))
            if w_new >= W - 4:
                continue

            piece = cv2.resize(crop, (w_new, h_new),
                               interpolation=cv2.INTER_AREA
                               if h_new < ch else cv2.INTER_LINEAR)

            y2 = int(round(yb * H))
            y1 = y2 - h_new
            if y1 < 0:
                continue
            x1 = random.randint(0, W - w_new - 1)
            x2 = x1 + w_new
            if not iou_free((x1, y1, x2, y2), boxes):
                continue

            region = canvas[y1:y2, x1:x2]
            piece = match_brightness(piece, region, a.bright)

            if a.blend == "seamless":
                m = (build_alpha(h_new, w_new, a.mask, a.feather) > 0.5)
                m = (m.astype(np.uint8) * 255)
                if m.sum() == 0:
                    continue
                try:
                    canvas = cv2.seamlessClone(
                        piece, canvas, m,
                        ((x1 + x2) // 2, (y1 + y2) // 2), cv2.NORMAL_CLONE)
                except cv2.error:
                    continue
            else:
                al = build_alpha(h_new, w_new, a.mask, a.feather)[..., None]
                canvas[y1:y2, x1:x2] = (
                    piece.astype(np.float32) * al
                    + region.astype(np.float32) * (1.0 - al)).astype(np.uint8)

            boxes.append((x1, y1, x2, y2))
            rows.append((T[w], ((x1 + x2) / 2.0 / W, (y1 + y2) / 2.0 / H,
                                w_new / float(W), h_new / float(H))))
            placed[w] += 1

        if not rows:
            continue

        stem = "cp_%05d" % made
        cv2.imwrite(os.path.join(out, "images", stem + ".jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(out, "labels", stem + ".txt"), "w") as f:
            for c, b in rows:
                f.write("%d %.6f %.6f %.6f %.6f\n" % (c, b[0], b[1], b[2], b[3]))
        if len(previews) < a.preview:
            v = canvas.copy()
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(v, (x1, y1), (x2, y2), (0, 255, 0), 2)
            previews.append(v)
        made += 1

    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# copy-paste 합성본. train 전용 — valid 로 보내지 마세요.\n")
        f.write("train: images\nval: images\n")
        f.write("nc: %d\nnames: [%s]\n" % (len(target), ", ".join(target)))

    print("\n" + "=" * 68)
    print("생성 %d장" % made)
    for w in want:
        print("   %-8s %d개 배치" % (w, placed[w]))

    if previews:
        cols = 4
        rows_n = (len(previews) + cols - 1) // cols
        th, tw = 240, 360
        sheet = np.full((rows_n * th, cols * tw, 3), 30, np.uint8)
        for i, v in enumerate(previews):
            r, c = divmod(i, cols)
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = \
                cv2.resize(v, (tw, th))
        pp = os.path.join(out, "preview.jpg")
        cv2.imwrite(pp, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("\n★ 대조표: %s" % pp)
        print("  반드시 열어서 확인하세요. 붙인 자리가 눈에 띄면 그 설정은 버리세요.")
        print("  - 경계 상자가 보인다      -> --mask ellipse, --feather 0.15")
        print("  - 밝기가 겉돈다           -> --blend seamless")
        print("  - 하늘/건물에 떠 있다     -> --y-range 로 구간을 좁히세요")
    return 0


if __name__ == "__main__":
    sys.exit(main())
