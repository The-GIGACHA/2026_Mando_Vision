#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remap_labels.py — 외부 YOLO 데이터셋을 우리 클래스 스키마로 변환합니다.

  배치 위치:  mando_vision_2026/tools/remap_labels.py

────────────────────────────────────────────────────────────────────────
왜 필요한가

Roboflow Universe 등에서 받은 데이터셋은 클래스 이름과 순서가 우리와
다릅니다. 그대로 업로드하면 Roboflow 가 새 클래스를 계속 만들어내서
클래스가 15개쯤 되고, 그 시점부터 정리가 불가능해집니다.

업로드 '전에' 라벨 txt 의 클래스 id 를 우리 스키마로 바꿔 두면
Roboflow 가 기존 클래스에 그대로 붙입니다.

────────────────────────────────────────────────────────────────────────
사용

  # 1) 먼저 무엇이 들어있는지 봅니다 (아무것도 바꾸지 않음)
  python3 tools/remap_labels.py datasets/universe_a --inspect

  # 2) 매핑 파일을 만듭니다 (--inspect 출력이 초안을 찍어줍니다)
  #    mapping.yaml:
  #      target: [green_left, green, left, red, yellow]
  #      map:
  #        red_left: left          # 이름이 달라도 같은 뜻이면 이렇게
  #        Red:      red
  #        arrow_green: green_left
  #        pedestrian_red: __DROP__   # 우리 스키마에 없으면 버립니다
  #
  # 3) 미리보기 (기본은 dry-run 입니다. 아무 파일도 안 바뀝니다)
  python3 tools/remap_labels.py datasets/universe_a --map mapping.yaml

  # 4) 실제 적용 — 원본은 건드리지 않고 새 폴더로 씁니다
  python3 tools/remap_labels.py datasets/universe_a --map mapping.yaml \\
      --out datasets/universe_a_remapped --apply

────────────────────────────────────────────────────────────────────────
결과물

  <out>/images/...       (원본 복사)
  <out>/labels/...       (클래스 id 를 target 순서로 바꾼 txt)
  <out>/data.yaml        (target 스키마)

  이 폴더의 images/ 와 labels/ 를 Roboflow 업로드 화면에 같이 드래그하면
  기존 클래스에 그대로 붙습니다.
"""

import argparse
import os
import shutil
import sys
from collections import Counter

DROP = "__DROP__"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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


def walk_labels(root):
    """(라벨파일 절대경로, root 기준 상대경로) 목록"""
    out = []
    for dp, _dn, fns in os.walk(root):
        if os.path.basename(dp) != "labels":
            continue
        for fn in fns:
            if fn.endswith(".txt") and fn != "classes.txt":
                p = os.path.join(dp, fn)
                out.append((p, os.path.relpath(p, root)))
    return sorted(out)


def walk_images(root):
    out = []
    for dp, _dn, fns in os.walk(root):
        if os.path.basename(dp) != "images":
            continue
        for fn in fns:
            if fn.lower().endswith(IMG_EXT):
                p = os.path.join(dp, fn)
                out.append((p, os.path.relpath(p, root)))
    return sorted(out)


def read_label(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            rows.append((cid, parts[1:]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="외부 데이터셋 폴더 (data.yaml 이 있는 곳)")
    ap.add_argument("--inspect", action="store_true",
                    help="클래스 분포만 보고 매핑 초안을 출력합니다")
    ap.add_argument("--map", dest="mapfile", default=None,
                    help="매핑 yaml (target, map)")
    ap.add_argument("--out", default=None, help="출력 폴더")
    ap.add_argument("--apply", action="store_true",
                    help="실제로 씁니다. 없으면 dry-run")
    ap.add_argument("--copy-images", dest="copy_images", default="copy",
                    choices=["copy", "symlink", "none"],
                    help="이미지 처리 방식 (기본 copy)")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    yml = os.path.join(root, "data.yaml")
    if not os.path.isfile(yml):
        sys.exit("data.yaml 이 없습니다: %s" % yml)
    src_names = names_of(load_yaml(yml))
    if not src_names:
        sys.exit("data.yaml 에 names 가 없습니다")

    labels = walk_labels(root)
    if not labels:
        sys.exit("labels/ 폴더를 찾지 못했습니다")

    cnt = Counter()
    for p, _ in labels:
        for cid, _r in read_label(p):
            cnt[cid] += 1

    # ── inspect ──────────────────────────────────────────────────
    if a.inspect or not a.mapfile:
        print("=" * 68)
        print("외부 데이터셋: %s" % root)
        print("이미지 %d장 / 라벨파일 %d개" % (len(walk_images(root)), len(labels)))
        print("=" * 68)
        print("클래스 (data.yaml 순서):")
        for i, n in enumerate(src_names):
            c = cnt.get(i, 0)
            flag = "" if c else "   (인스턴스 0개)"
            print("   %2d  %-24s %6d%s" % (i, n, c, flag))
        unknown = [i for i in cnt if i >= len(src_names)]
        if unknown:
            print("\n! data.yaml 에 없는 클래스 id 가 라벨에 있습니다: %s" % unknown)

        print("\n" + "-" * 68)
        print("매핑 파일 초안입니다. 오른쪽을 우리 클래스명으로 고치고,")
        print("필요 없는 것은 %s 로 두세요." % DROP)
        print("-" * 68)
        print("target: [green_left, green, left, red, yellow]   # 우리 스키마 (순서 중요)")
        print("map:")
        for i, n in enumerate(src_names):
            print("  %-24s %s      # %d개" % (n + ":", DROP, cnt.get(i, 0)))
        if not a.mapfile:
            print("\n(--map 이 없어 검사만 했습니다)")
        return 0

    # ── 매핑 적용 ────────────────────────────────────────────────
    mp = load_yaml(a.mapfile)
    target = list(mp.get("target") or [])
    table = dict(mp.get("map") or {})
    if not target:
        sys.exit("매핑 파일에 target 이 없습니다")

    tindex = {n: i for i, n in enumerate(target)}
    missing = [n for n in src_names if n not in table]
    if missing:
        sys.exit("매핑에 없는 클래스가 있습니다 (%s). "
                 "%s 로라도 명시하세요." % (", ".join(missing), DROP))

    bad = [v for v in table.values() if v != DROP and v not in tindex]
    if bad:
        sys.exit("target 에 없는 대상으로 매핑했습니다: %s" % ", ".join(sorted(set(bad))))

    # src id -> dst id (또는 None)
    remap = {}
    for i, n in enumerate(src_names):
        dst = table[n]
        remap[i] = None if dst == DROP else tindex[dst]

    print("=" * 68)
    print("매핑")
    print("=" * 68)
    moved = Counter()
    dropped = 0
    for i, n in enumerate(src_names):
        c = cnt.get(i, 0)
        if remap[i] is None:
            print("   %-24s -> 버림           %6d" % (n, c))
            dropped += c
        else:
            t = target[remap[i]]
            print("   %-24s -> %-16s %6d" % (n, "%d:%s" % (remap[i], t), c))
            moved[remap[i]] += c

    print("\n변환 후 클래스별 인스턴스")
    for i, n in enumerate(target):
        c = moved.get(i, 0)
        mark = "   ← 이 데이터셋에는 없음" if c == 0 else ""
        print("   %d: %-20s %6d%s" % (i, n, c, mark))
    print("\n버려지는 인스턴스: %d개" % dropped)

    if not a.apply:
        print("\n[dry-run] 실제로 쓰려면 --out <폴더> --apply 를 붙이세요.")
        return 0
    if not a.out:
        sys.exit("--apply 에는 --out 이 필요합니다")

    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit("출력 폴더가 비어있지 않습니다: %s" % out)

    n_lab = n_row = 0
    for p, rel in labels:
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        keep = []
        for cid, r in read_label(p):
            new = remap.get(cid)
            if new is None:
                continue
            keep.append("%d %s" % (new, " ".join(r)))
            n_row += 1
        with open(dst, "w", encoding="utf-8") as f:
            f.write("\n".join(keep) + ("\n" if keep else ""))
        n_lab += 1

    n_img = 0
    if a.copy_images != "none":
        for p, rel in walk_images(root):
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if a.copy_images == "symlink":
                if not os.path.exists(dst):
                    os.symlink(p, dst)
            else:
                shutil.copy2(p, dst)
            n_img += 1

    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("train: train/images\nval: valid/images\n")
        f.write("nc: %d\n" % len(target))
        f.write("names: [%s]\n" % ", ".join(target))

    print("\n완료: 라벨 %d개 (%d줄), 이미지 %d장 -> %s" % (n_lab, n_row, n_img, out))
    print("이제 %s/images 와 %s/labels 를 Roboflow 업로드에 같이 드래그하세요."
          % (out, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
