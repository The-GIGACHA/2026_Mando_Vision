#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_signcar.py — 신호제어차량 데이터셋을 좌/우 4클래스로 변환합니다.

  배치 위치:  mando_vision_2026/tools/convert_signcar.py

────────────────────────────────────────────────────────────────────────
무엇을 하는가

외부 데이터셋(예: universe/s-workspace-flru6/mando-u0lff)은
  truck / green_sign / red_sign
처럼 패널의 좌·우를 구분하지 않습니다. 우리 스키마는
  left_go / left_X / right_go / right_X
로 좌·우가 의미를 갖습니다(어느 차로가 열렸는지).

같은 이미지 안에서 차량 박스 중심과 패널 박스 중심을 비교하면
좌·우를 기계적으로 판정할 수 있습니다.

    패널 x중심 <  차량 x중심 - 여유  →  left_*
    패널 x중심 >  차량 x중심 + 여유  →  right_*
    그 사이(여유 안)                →  판정 불가

    green_sign(↓) → *_go        red_sign(X) → *_X

────────────────────────────────────────────────────────────────────────
★ 버리는 이미지가 왜 필요한가

좌·우를 못 정한 패널이 있는 이미지를 '패널 라벨 없이' 남기면,
모델은 그 패널을 **배경으로 학습합니다.** 미검출을 직접 가르치는 셈이라
안 쓰느니만 못합니다. 그래서 판정 불가 패널이 하나라도 있으면
이미지를 통째로 제외합니다.

차량만 있고 패널이 없는 이미지는 그대로 씁니다(차량 검출 학습에 유용).

────────────────────────────────────────────────────────────────────────
측면 뷰 문제

차량을 뒤에서 보면 좌·우 판정이 맞지만, 옆에서 보면 뒤집힐 수 있습니다.
후면 뷰는 박스가 대체로 정사각형에 가깝고 측면 뷰는 가로로 깁니다.
--inspect 로 종횡비 분포를 보고 --max-aspect 로 잘라내세요.

────────────────────────────────────────────────────────────────────────
사용

  # 1) 구성과 종횡비 분포 확인 (아무것도 안 바꿉니다)
  python3 tools/convert_signcar.py datasets/mando_u0lff --inspect

  # 2) 미리보기
  python3 tools/convert_signcar.py datasets/mando_u0lff --max-aspect 1.8

  # 3) 적용
  python3 tools/convert_signcar.py datasets/mando_u0lff \\
      --out datasets/signcar_4cls --max-aspect 1.8 --apply

  결과 폴더의 images/ 와 labels/ 를 Roboflow 업로드에 같이 드래그하세요.
"""

import argparse
import os
import shutil
import sys
from collections import Counter

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 우리 obstacle 모델 스키마 (순서 = 클래스 id)
DEFAULT_TARGET = ["T870", "kid", "traffic_car",
                  "left_go", "left_X", "right_go", "right_X"]


def load_yaml(p):
    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml 이 필요합니다:  pip install pyyaml --break-system-packages")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def names_of(data):
    n = data.get("names")
    if isinstance(n, dict):
        return [n[k] for k in sorted(n, key=lambda z: int(z))]
    return list(n or [])


def walk(root, kind):
    out = []
    for dp, _dn, fns in os.walk(root):
        if os.path.basename(dp) != kind:
            continue
        for fn in fns:
            ok = fn.endswith(".txt") and fn != "classes.txt" if kind == "labels" \
                else fn.lower().endswith(IMG_EXT)
            if ok:
                p = os.path.join(dp, fn)
                out.append((p, os.path.relpath(p, root)))
    return sorted(out)


def read_label(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.split()
            if len(s) < 5:
                continue
            try:
                rows.append((int(float(s[0])), [float(v) for v in s[1:5]],
                             s[5:]))
            except ValueError:
                continue
    return rows


def image_for(label_path, root):
    """labels/xxx.txt 에 대응하는 images/xxx.* 를 찾습니다."""
    d = os.path.dirname(label_path)
    stem = os.path.splitext(os.path.basename(label_path))[0]
    img_dir = os.path.join(os.path.dirname(d), "images")
    for e in IMG_EXT:
        p = os.path.join(img_dir, stem + e)
        if os.path.isfile(p):
            return p
    return None


def image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size          # (W, H)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--truck-name", default="truck")
    ap.add_argument("--go-name", default="green_sign", help="↓ (차로 개방)")
    ap.add_argument("--x-name", default="red_sign", help="X (차로 폐쇄)")
    ap.add_argument("--target", default=",".join(DEFAULT_TARGET),
                    help="출력 클래스 순서 (쉼표 구분)")
    ap.add_argument("--deadband", type=float, default=0.05,
                    help="차량 폭 대비 중앙 무판정 구간 (기본 0.05)")
    ap.add_argument("--max-aspect", type=float, default=0.0,
                    help="차량 박스 가로/세로 상한. 0 이면 미적용")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--copy-images", dest="copy_images", default="copy",
                    choices=["copy", "symlink", "none"])
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    yml = os.path.join(root, "data.yaml")
    if not os.path.isfile(yml):
        sys.exit("data.yaml 이 없습니다: %s" % yml)
    src = names_of(load_yaml(yml))
    idx = {n: i for i, n in enumerate(src)}
    for need in (a.truck_name, a.go_name, a.x_name):
        if need not in idx:
            sys.exit("클래스 '%s' 가 data.yaml 에 없습니다. 있는 클래스: %s"
                     % (need, src))
    I_TRUCK, I_GO, I_X = idx[a.truck_name], idx[a.go_name], idx[a.x_name]

    target = [s.strip() for s in a.target.split(",") if s.strip()]
    T = {n: i for i, n in enumerate(target)}
    for need in ("traffic_car", "left_go", "left_X", "right_go", "right_X"):
        if need not in T:
            sys.exit("target 에 '%s' 가 없습니다" % need)

    labels = walk(root, "labels")
    if not labels:
        sys.exit("labels/ 를 찾지 못했습니다")

    aspects = []
    out_rows = {}          # 상대경로 -> [(cls, [x,y,w,h])]
    stat = Counter()
    drop_reason = Counter()

    for lp, rel in labels:
        rows = read_label(lp)
        trucks = [b for c, b, _ in rows if c == I_TRUCK]
        signs = [(c, b) for c, b, _ in rows if c in (I_GO, I_X)]
        other = [(c, b) for c, b, _ in rows if c not in (I_TRUCK, I_GO, I_X)]

        # 종횡비 (픽셀 기준). 이미지 크기를 못 읽으면 정규화 값 그대로
        ip = image_for(lp, root)
        wh = image_size(ip) if ip else None
        for b in trucks:
            ar = (b[2] / max(b[3], 1e-9))
            if wh:
                ar *= (wh[0] / float(wh[1]))
            aspects.append(ar)

        if a.inspect:
            stat["image"] += 1
            stat["truck"] += len(trucks)
            stat["sign"] += len(signs)
            if signs and not trucks:
                drop_reason["패널은 있는데 차량 박스 없음"] += 1
            continue

        keep = []
        bad = None

        # 측면 뷰 제외
        if a.max_aspect > 0 and trucks:
            for b in trucks:
                ar = b[2] / max(b[3], 1e-9)
                if wh:
                    ar *= (wh[0] / float(wh[1]))
                if ar > a.max_aspect:
                    bad = "측면 뷰로 판단 (종횡비 %.2f)" % ar
                    break

        if bad is None and signs and not trucks:
            bad = "패널은 있는데 차량 박스 없음"

        if bad is None:
            for c, b in signs:
                sx = b[0]
                # 패널을 포함하는 차량 우선, 없으면 x 로 가장 가까운 차량
                host = None
                for t in trucks:
                    if (t[0] - t[2] / 2 - 0.02 <= sx <= t[0] + t[2] / 2 + 0.02
                            and t[1] - t[3] / 2 - 0.02 <= b[1] <= t[1] + t[3] / 2 + 0.02):
                        host = t
                        break
                if host is None:
                    host = min(trucks, key=lambda t: abs(t[0] - sx))

                dead = a.deadband * host[2]
                if sx < host[0] - dead:
                    side = "left"
                elif sx > host[0] + dead:
                    side = "right"
                else:
                    bad = "패널이 차량 중앙 무판정 구간에 있음"
                    break
                name = "%s_%s" % (side, "go" if c == I_GO else "X")
                keep.append((T[name], b))

        stat["image"] += 1
        if bad:
            drop_reason[bad] += 1
            stat["image_dropped"] += 1
            continue

        for t in trucks:
            keep.append((T["traffic_car"], t))
        for c, b in other:
            pass        # 우리 스키마에 없는 클래스는 버립니다

        out_rows[rel] = keep
        stat["image_kept"] += 1
        for c, _ in keep:
            stat["cls_%d" % c] += 1

    # ── 리포트 ───────────────────────────────────────────────────
    print("=" * 70)
    print("입력: %s" % root)
    print("클래스: %s" % src)
    print("=" * 70)

    if aspects:
        aspects.sort()
        def q(p):
            return aspects[min(int(len(aspects) * p), len(aspects) - 1)]
        print("차량 박스 종횡비 (가로/세로, 픽셀 기준)")
        print("   10%% %.2f   25%% %.2f   중앙 %.2f   75%% %.2f   90%% %.2f   최대 %.2f"
              % (q(.10), q(.25), q(.50), q(.75), q(.90), aspects[-1]))
        print("   → 후면 뷰는 중앙값 근처, 측면 뷰는 오른쪽 꼬리입니다.")
        print("     --max-aspect 를 75~90% 값 근처로 잡아보세요.")

    if a.inspect:
        print("\n이미지 %d장 / 차량 %d개 / 패널 %d개"
              % (stat["image"], stat["truck"], stat["sign"]))
        for k, v in drop_reason.items():
            print("   ! %s: %d장" % (k, v))
        print("\n(--inspect 라 변환은 하지 않았습니다)")
        return 0

    print("\n변환 결과")
    print("   이미지 %d장 중 사용 %d장 / 제외 %d장"
          % (stat["image"], stat["image_kept"], stat["image_dropped"]))
    for k, v in drop_reason.most_common():
        print("      제외: %-32s %d장" % (k, v))
    print("\n   클래스별 인스턴스")
    for i, n in enumerate(target):
        c = stat.get("cls_%d" % i, 0)
        mark = "   (이 데이터셋에는 없음 — 우리 데이터에서 채웁니다)" if c == 0 else ""
        print("      %d: %-12s %6d%s" % (i, n, c, mark))

    lg, rg = stat.get("cls_%d" % T["left_go"], 0), stat.get("cls_%d" % T["right_go"], 0)
    lx, rx = stat.get("cls_%d" % T["left_X"], 0), stat.get("cls_%d" % T["right_X"], 0)
    if lg + rg + lx + rx:
        print("\n   좌/우 균형  left %d : right %d" % (lg + lx, rg + rx))
        if min(lg + lx, rg + rx) < 0.25 * max(lg + lx, rg + rx):
            print("   ! 한쪽으로 크게 치우쳤습니다. 적은 쪽 검출이 약해집니다.")

    if not a.apply:
        print("\n[dry-run] 실제로 쓰려면 --out <폴더> --apply 를 붙이세요.")
        return 0
    if not a.out:
        sys.exit("--apply 에는 --out 이 필요합니다")

    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit("출력 폴더가 비어있지 않습니다: %s" % out)

    n_img = 0
    for rel, rows in out_rows.items():
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            for c, b in rows:
                f.write("%d %.6f %.6f %.6f %.6f\n" % (c, b[0], b[1], b[2], b[3]))
        lp = os.path.join(root, rel)
        ip = image_for(lp, root)
        if ip and a.copy_images != "none":
            rel_img = os.path.relpath(ip, root)
            dsti = os.path.join(out, rel_img)
            os.makedirs(os.path.dirname(dsti), exist_ok=True)
            if a.copy_images == "symlink":
                if not os.path.exists(dsti):
                    os.symlink(ip, dsti)
            else:
                shutil.copy2(ip, dsti)
            n_img += 1

    # ── data.yaml 은 '실제로 만들어진' 구조에 맞춰 씁니다 ────────────
    #    (입력 zip 이 평평하거나 상위 폴더가 하나 더 있는 경우가 흔합니다)
    splits = {}
    for dp, _dn, _fns in os.walk(out):
        if os.path.basename(dp).lower() != "images":
            continue
        if not os.path.isdir(os.path.join(os.path.dirname(dp), "labels")):
            continue
        rel = os.path.relpath(dp, out).replace(os.sep, "/")
        key = None
        for part in rel.lower().split("/"):
            if part in ("train", "training"):
                key = "train"
            elif part in ("valid", "val", "validation"):
                key = "val"
            elif part == "test":
                key = "test"
        splits.setdefault(key or "train", rel)

    note = ""
    if "val" not in splits:
        splits["val"] = splits.get("train", "images")
        note = ("# ! 이 데이터셋에는 valid 분할이 없어 val 을 train 과 같게 두었습니다.\n"
                "#   Roboflow 에 올려 분할을 지정하거나, 우리 촬영분을 val 로 넣으세요.\n")

    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(note)
        f.write("train: %s\nval: %s\n" % (splits.get("train", "images"),
                                          splits["val"]))
        if "test" in splits:
            f.write("test: %s\n" % splits["test"])
        f.write("nc: %d\nnames: [%s]\n" % (len(target), ", ".join(target)))

    print("\n완료: 라벨 %d개 / 이미지 %d장 -> %s" % (len(out_rows), n_img, out))
    print("검사:  python3 tools/check_dataset.py %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
