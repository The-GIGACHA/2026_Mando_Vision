#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_dataset.py — 학습 걸기 전 데이터셋 검사.

  배치 위치:  mando_vision_2026/tools/check_dataset.py

────────────────────────────────────────────────────────────────────────
학습은 한 번 돌리면 30~60분입니다. 잘못된 데이터로 돌리면 그 시간과
그 뒤의 검증 시간까지 통째로 날아갑니다. 이 스크립트는 그 전에
"돌려도 되는 데이터인가"를 1초 만에 답합니다.

검사 항목 — 각각 실제로 대회를 망칠 수 있는 것들입니다

  ① 클래스 순서       원본 모델과 다르면 파인튜닝이 엉뚱한 클래스를 덮어씁니다
                      (작년 sign.pt / traffic_sign.pt 뒤바뀜과 같은 계열)
  ② 클래스별 인스턴스 0개면 그 클래스를 잊어버립니다 (catastrophic forgetting)
  ③ train/val 누수    영상 프레임을 무작위 분할하면 같은 장면이 양쪽에 갑니다.
                      val mAP 0.95 인데 현장에서 0인 전형적 원인
  ④ 박스 크기 분포    너무 작은 박스는 학습을 방해합니다
  ⑤ 라벨 파일 무결성  좌표 범위, 클래스 id 범위, 이미지-라벨 짝

────────────────────────────────────────────────────────────────────────
사용

  python3 tools/check_dataset.py datasets/traffic_light

  # 파인튜닝 대상 모델과 클래스 순서 비교까지
  python3 tools/check_dataset.py datasets/traffic_light \
      --against weights/traffic_light_20250918.pt

  # 누수 판정 기준을 파일명 앞 N글자로 (기본 8)
  python3 tools/check_dataset.py datasets/obstacle --stem-len 12
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_yaml(path):
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        sys.exit("pyyaml 이 필요합니다:  pip install pyyaml --break-system-packages")


def find_split_dirs(root, data):
    """data.yaml 의 train/val/test 경로를 실제 디렉터리로 해석합니다.

    Roboflow 는 'train: ../train/images' 처럼 내보내는데, 이게 어디를
    기준으로 하는지가 배포 방식마다 달라집니다. 흔한 경우를 모두 시도합니다."""
    base = data.get("path")
    roots = [root]
    if base:
        roots.insert(0, base if os.path.isabs(base)
                     else os.path.normpath(os.path.join(root, base)))

    fallback = {"train": ["train"], "val": ["valid", "val"],
                "test": ["test"]}

    out = {}
    for key in ("train", "val", "test"):
        v = data.get(key)
        cands = []
        if v:
            cands = [v] if isinstance(v, str) else list(v)
        tries = []
        for c in cands:
            if os.path.isabs(c):
                tries.append(c)
                continue
            stripped = re.sub(r"^(\.\./)+", "", c)      # 앞의 ../ 제거
            for r in roots:
                tries.append(os.path.normpath(os.path.join(r, c)))
                tries.append(os.path.normpath(os.path.join(r, stripped)))
        for name in fallback[key]:                      # 마지막 수단: 관례적 배치
            for r in roots:
                tries.append(os.path.join(r, name, "images"))
                tries.append(os.path.join(r, name))
        for p in tries:
            if os.path.isdir(p) and any(
                    f.lower().endswith(IMG_EXT) for f in os.listdir(p)):
                out[key] = p
                break

    if not out:
        out = discover_recursive(root)
    return out


def discover_recursive(root):
    """data.yaml 경로가 실제 구조와 안 맞을 때의 최후 수단.

    zip 안에 상위 폴더가 하나 더 있거나(datasets/x/x/train/images),
    분할 없이 평평하게 풀린 경우(datasets/x/images)를 모두 잡습니다.
    'labels' 형제 폴더가 있는 'images' 폴더를 전부 찾아,
    경로에 train/valid/test 가 들어있으면 그 분할로, 없으면 train 으로 봅니다."""
    found = {}
    for dp, _dn, _fns in os.walk(root):
        if os.path.basename(dp).lower() != "images":
            continue
        if not os.path.isdir(os.path.join(os.path.dirname(dp), "labels")):
            continue
        try:
            if not any(f.lower().endswith(IMG_EXT) for f in os.listdir(dp)):
                continue
        except OSError:
            continue
        key = None
        for part in os.path.relpath(dp, root).lower().split(os.sep):
            if part in ("train", "training"):
                key = "train"
            elif part in ("valid", "val", "validation"):
                key = "val"
            elif part == "test":
                key = "test"
        found.setdefault(key or "train", dp)
    if found:
        print("! data.yaml 의 경로가 실제 구조와 달라 직접 찾았습니다:")
        for k in ("train", "val", "test"):
            if k in found:
                print("    %-5s -> %s" % (k, os.path.relpath(found[k], root)))
        if "val" not in found:
            print("    (val 이 없습니다 — Roboflow 에 올려 분할을 지정하거나")
            print("     data.yaml 의 val 경로를 맞춰 주세요)")
        print()
    return found


def labels_dir_for(img_dir):
    """.../images -> .../labels"""
    if os.path.basename(img_dir) == "images":
        return os.path.join(os.path.dirname(img_dir), "labels")
    return os.path.join(img_dir, "..", "labels")


def scan_split(img_dir):
    lab_dir = labels_dir_for(img_dir)
    imgs = [f for f in sorted(os.listdir(img_dir))
            if f.lower().endswith(IMG_EXT)]
    counts = Counter()
    boxes = []            # (cls, w_norm, h_norm)
    missing = []
    empty = []
    bad = []

    for fn in imgs:
        stem = os.path.splitext(fn)[0]
        lp = os.path.join(lab_dir, stem + ".txt")
        if not os.path.isfile(lp):
            missing.append(fn)
            continue
        n = 0
        with open(lp, "r", encoding="utf-8", errors="replace") as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) < 5:
                    bad.append("%s:%d 필드 부족" % (stem, ln))
                    continue
                try:
                    c = int(float(parts[0]))
                    x, y, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    bad.append("%s:%d 숫자 아님" % (stem, ln))
                    continue
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
                        and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    bad.append("%s:%d 좌표 범위 벗어남" % (stem, ln))
                    continue
                counts[c] += 1
                boxes.append((c, w, h))
                n += 1
        if n == 0:
            empty.append(fn)

    return dict(n_img=len(imgs), counts=counts, boxes=boxes,
                missing=missing, empty=empty, bad=bad,
                stems=[os.path.splitext(f)[0] for f in imgs])


def prefix_key(stem, n):
    """파일명에서 촬영 출처를 추정하는 키.

    Roboflow 는 업로드 파일명 뒤에 해시를 붙입니다
    (예: yongin_f000120_png.rf.a1b2c3....jpg)
    앞쪽 N 글자가 같으면 같은 영상/구간에서 나온 프레임으로 봅니다."""
    s = re.split(r"\.rf\.", stem)[0]
    s = re.sub(r"[-_]?\d+$", "", s)
    return s[:n] if len(s) > n else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="데이터셋 폴더 (data.yaml 이 있는 곳)")
    ap.add_argument("--against", default=None,
                    help="파인튜닝 대상 .pt — 클래스 순서를 비교합니다")
    ap.add_argument("--stem-len", type=int, default=8,
                    help="누수 판정용 파일명 접두 길이 (기본 8)")
    ap.add_argument("--min-per-class", type=int, default=100,
                    help="클래스별 최소 인스턴스 경고 기준")
    ap.add_argument("--tiny-px", type=int, default=12,
                    help="이 픽셀(640 기준) 미만 박스를 '너무 작음'으로 셈")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    yml = os.path.join(root, "data.yaml")
    if not os.path.isfile(yml):
        sys.exit("data.yaml 이 없습니다: %s" % yml)
    data = load_yaml(yml)

    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda z: int(z))]
    names = list(names or [])
    nc = int(data.get("nc", len(names)))

    print("=" * 72)
    print("데이터셋: %s" % root)
    print("=" * 72)
    print("클래스 %d개 (data.yaml 순서 그대로):" % nc)
    for i, n in enumerate(names):
        print("   %d: %s" % (i, n))

    problems = []
    warns = []

    # ── ① 클래스 순서 비교 ────────────────────────────────────────
    if a.against:
        try:
            from ultralytics import YOLO
            ref = YOLO(a.against).names
            ref = [ref[k] for k in sorted(ref)]
            print("\n① 클래스 순서 비교  (기준: %s)" % os.path.basename(a.against))
            if ref == names:
                print("   일치 — 그대로 파인튜닝하면 됩니다")
            else:
                print("   원본 : %s" % ref)
                print("   현재 : %s" % names)
                if sorted(ref) == sorted(names):
                    print("   → 이름은 같은데 순서가 다릅니다.")
                    print("      data.yaml 의 names 를 원본 순서로 고치세요:")
                    print("      names: %s" % ref)
                    problems.append("클래스 순서 불일치 (치명적)")
                else:
                    print("   → 클래스 집합 자체가 다릅니다. 신규 학습이라면 정상입니다.")
                    warns.append("클래스 집합이 원본과 다름")
        except Exception as e:
            print("\n① 클래스 순서 비교 실패: %s" % e)

    # ── 분할별 스캔 ──────────────────────────────────────────────
    splits = find_split_dirs(root, data)
    if not splits:
        sys.exit("train/val 경로를 찾지 못했습니다. data.yaml 을 확인하세요.")

    res = {}
    print("\n② 분할별 이미지 / 인스턴스")
    for k in ("train", "val", "test"):
        if k not in splits:
            continue
        r = scan_split(splits[k])
        res[k] = r
        print("   %-6s 이미지 %5d   인스턴스 %6d"
              % (k, r["n_img"], sum(r["counts"].values())))
        if r["missing"]:
            print("          ! 라벨 파일 없음 %d장" % len(r["missing"]))
            warns.append("%s: 라벨 없는 이미지 %d장" % (k, len(r["missing"])))
        if r["empty"]:
            print("          - 배경 이미지(라벨 0개) %d장" % len(r["empty"]))
        if r["bad"]:
            print("          ! 잘못된 라벨 줄 %d개  예) %s"
                  % (len(r["bad"]), r["bad"][0]))
            problems.append("%s: 손상된 라벨 %d줄" % (k, len(r["bad"])))

    # ── 클래스별 분포 ────────────────────────────────────────────
    print("\n③ 클래스별 인스턴스")
    hdr = "   %-18s" % "클래스"
    for k in res:
        hdr += "%8s" % k
    print(hdr + "%9s" % "합계")
    for i in range(nc):
        nm = names[i] if i < len(names) else str(i)
        row = "   %-18s" % nm
        tot = 0
        for k in res:
            c = res[k]["counts"].get(i, 0)
            tot += c
            row += "%8d" % c
        row += "%9d" % tot
        if tot == 0:
            row += "   ★ 인스턴스 0개 — 이 클래스는 잊혀집니다"
            problems.append("클래스 '%s' 인스턴스 0개" % nm)
        elif tot < a.min_per_class:
            row += "   ! %d개 미만" % a.min_per_class
            warns.append("클래스 '%s' 인스턴스 %d개 (부족)" % (nm, tot))
        if "val" in res and res["val"]["counts"].get(i, 0) == 0 and tot > 0:
            row += "   ! val 에 없음(성능 측정 불가)"
            warns.append("클래스 '%s' 가 val 에 없음" % nm)
        print(row)

    # ── ④ train/val 누수 ─────────────────────────────────────────
    if "train" in res and "val" in res:
        tr = {prefix_key(s, a.stem_len) for s in res["train"]["stems"]}
        va_keys = [prefix_key(s, a.stem_len) for s in res["val"]["stems"]]
        shared = sorted({k for k in va_keys if k in tr})
        overlap = sum(1 for k in va_keys if k in tr)
        pct = 100.0 * overlap / max(len(va_keys), 1)
        print("\n④ train/val 누수 점검  (파일명 앞 %d글자 기준)" % a.stem_len)
        print("   val 이미지 %d장 중 train 과 같은 출처로 보이는 것: %d장 (%.0f%%)"
              % (len(va_keys), overlap, pct))
        if pct >= 50:
            print("   ★ 같은 영상이 train 과 val 양쪽에 있습니다.")
            print("      val mAP 가 높게 나와도 현장 성능과 무관합니다.")
            print("      Roboflow 에서 특정 영상/회차를 통째로 val 로 옮기세요.")
            print("      겹치는 접두: %s" % ", ".join(shared[:8]))
            problems.append("train/val 누수 %.0f%%" % pct)
        elif pct > 0:
            print("   일부 겹침 — 접두: %s" % ", ".join(shared[:8]))
            warns.append("train/val 일부 겹침 %.0f%%" % pct)
        else:
            print("   겹침 없음 — 정상")

    # ── ⑤ 박스 크기 ──────────────────────────────────────────────
    print("\n⑤ 박스 크기 (640px 기준 환산)")
    per_cls = defaultdict(list)
    for k in res:
        for c, w, h in res[k]["boxes"]:
            per_cls[c].append(min(w, h) * 640.0)
    for i in range(nc):
        v = per_cls.get(i, [])
        if not v:
            continue
        v.sort()
        med = v[len(v) // 2]
        p10 = v[int(len(v) * 0.10)]
        tiny = sum(1 for x in v if x < a.tiny_px)
        nm = names[i] if i < len(names) else str(i)
        line = ("   %-18s 중앙 %5.1f px   하위10%% %5.1f px   %dpx미만 %d개(%.0f%%)"
                % (nm, med, p10, a.tiny_px, tiny, 100.0 * tiny / len(v)))
        if tiny > 0.25 * len(v):
            line += "  ! 작은 박스가 많습니다"
            warns.append("클래스 '%s' 작은 박스 %.0f%%" % (nm, 100.0 * tiny / len(v)))
        print(line)

    # ── 결론 ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    if problems:
        print("★ 학습 전에 고쳐야 합니다")
        for p in problems:
            print("   - %s" % p)
    if warns:
        print("! 확인해 보세요")
        for w in warns:
            print("   - %s" % w)
    if not problems and not warns:
        print("문제 없음 — 학습 진행하세요")
    print("=" * 72)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
