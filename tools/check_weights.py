#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_weights.py — 학습된 .pt 가중치가 실제로 로드되는지, 클래스 매핑이
                   작년 코드의 하드코딩과 일치하는지 검사합니다.

GPU 없어도 됩니다. 맥북 CPU에서 그대로 돌아갑니다.

사용:
    python3 check_weights.py <weights 폴더>
    python3 check_weights.py <weights 폴더>/lane.pt <weights 폴더>/cone.pt
    python3 check_weights.py .              # 현재 폴더에서 *.pt 전부

검사 항목:
  1. LFS 포인터인지 (zip으로 받으면 133바이트짜리 텍스트가 들어 있습니다)
  2. TorchScript(YOLOPv2) / Ultralytics(YOLOv12) 자동 판별
  3. 실제 forward 1회 — 로드만 되고 추론이 깨지는 경우를 잡습니다
  4. 클래스 이름·순서를 작년 코드의 하드코딩 매핑과 대조
"""

import sys
import os
import glob
import re
import traceback

# ── 작년 코드에 하드코딩되어 있던 매핑 (대조용) ────────────────────────
#   traffic_light_detector.py L109~129
LEGACY_TRAFFIC_LIGHT = {
    0: "green_left", 1: "green", 2: "red_left", 3: "red", 4: "yellow",
}
#   traffic_sign_fusion.cpp initializeSignClasses()
LEGACY_SIGN = {
    0: "stop", 1: "yield", 2: "speed_limit", 3: "no_entry", 4: "turn_left",
    5: "turn_right", 6: "straight", 7: "pedestrian_crossing",
    8: "parking", 9: "no_parking",
}
#   gigacha_sensor_fusion.cpp — 카메라 위치로 하드코딩
LEGACY_CONE = {0: "blue_cone", 1: "yellow_cone"}

LEGACY_BY_NAME = {
    "traffic_light": LEGACY_TRAFFIC_LIGHT,
    "traffic_sign":  LEGACY_SIGN,
    "sign":          LEGACY_CONE,     # README상 sign.pt = 콘 검출
    "cone":          LEGACY_CONE,
}

G, R, Y, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[94m", "\033[0m"


def is_lfs_pointer(path):
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs")
    except Exception:
        return False


def try_torchscript(path, torch):
    """YOLOPv2는 torch.jit 로 저장된 TorchScript 모델입니다."""
    m = torch.jit.load(path, map_location="cpu")
    m.eval()
    x = torch.zeros(1, 3, 384, 640)          # YOLOPv2 표준 입력 (16:9 letterbox)
    with torch.no_grad():
        out = m(x)

    info = {"type": "TorchScript (YOLOPv2 계열)"}
    try:
        (pred, anchor_grid), seg, ll = out
        # pred 는 텐서가 아니라 3개 텐서의 '리스트'입니다.
        # 객체 검출까지 쓰려면 utils.split_for_trace_model 에 넘기세요.
        if isinstance(pred, (list, tuple)):
            info["det"] = "%d개 head: %s" % (
                len(pred), [tuple(t.shape) for t in pred])
        else:
            info["det"] = str(tuple(pred.shape))
        info["drivable_area"] = str(tuple(seg.shape))
        info["lane_line"] = str(tuple(ll.shape))
        # utils.py 의 seg[:, :, 12:372, :] 크롭이 유효한지
        h = seg.shape[2]
        info["crop_ok"] = (h == 384)
    except Exception:
        info["outputs"] = "예상과 다른 출력 구조: %s" % type(out)
    return info


def try_ultralytics(path):
    from ultralytics import YOLO
    m = YOLO(path)
    info = {
        "type": "Ultralytics (%s)" % getattr(m, "task", "?"),
        "names": dict(m.names),
        "nc": len(m.names),
    }
    # 실제 추론 1회 — 로드만 되고 forward가 깨지는 경우를 잡습니다
    try:
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        m.predict(dummy, verbose=False, device="cpu")
        info["forward"] = True
    except Exception as e:
        info["forward"] = False
        info["forward_err"] = str(e)[:200]
    return info


_PREFIX = re.compile(r"^\d+_")


def _norm(n):
    """'0_green_left' -> 'green_left'. 접두어는 원본 통합 데이터셋 인덱스."""
    return _PREFIX.sub("", str(n)).strip()


def compare_legacy(stem, names):
    names = {int(k): _norm(v) for k, v in names.items()}
    """파일명으로 작년 하드코딩 매핑을 찾아 대조합니다."""
    legacy = None
    for key, mapping in LEGACY_BY_NAME.items():
        if key in stem.lower():
            legacy, legacy_key = mapping, key
            break
    if legacy is None:
        return None

    lines = ["  %s작년 코드 하드코딩 매핑과 대조 (%s)%s" % (B, legacy_key, X)]
    mismatch = False
    for i in sorted(set(list(legacy.keys()) + list(names.keys()))):
        old = legacy.get(i, "—")
        new = names.get(i, "—")
        if old == new:
            lines.append("    %2d  %-22s %s== 일치%s" % (i, new, G, X))
        else:
            mismatch = True
            lines.append("    %2d  실제:%-16s 작년코드:%-16s %s← 불일치%s"
                         % (i, new, old, R, X))
    if mismatch:
        lines.append("")
        lines.append("  %s★ 클래스 순서가 다릅니다. 작년 코드를 그대로 쓰면%s" % (R, X))
        lines.append("  %s  잘못된 판정이 나갑니다 (예: 빨간불에 출발).%s" % (R, X))
        lines.append("  %s  해당 노드의 클래스 매핑을 실제 names 기준으로 고치세요.%s" % (R, X))
    return "\n".join(lines)


def check_one(path, torch):
    stem = os.path.splitext(os.path.basename(path))[0]
    size = os.path.getsize(path)
    print("\n" + "=" * 72)
    print("%s  (%.1f MB)" % (path, size / 1e6))
    print("=" * 72)

    if is_lfs_pointer(path):
        print("%s✗ Git LFS 포인터입니다 — 실제 가중치가 아닙니다.%s" % (R, X))
        print("  zip 다운로드로는 LFS 파일이 안 옵니다. git clone 후:")
        print("      git lfs install && git lfs pull")
        return "LFS_POINTER"

    # 1) TorchScript 먼저 시도
    try:
        info = try_torchscript(path, torch)
        print("%s✓ 로드 성공%s — %s" % (G, X, info.pop("type")))
        for k, v in info.items():
            print("    %-16s %s" % (k, v))
        if info.get("crop_ok") is False:
            print("  %s! utils.py 의 seg[:,:,12:372,:] 크롭은 384 높이 전제입니다.%s"
                  % (Y, X))
        return "TORCHSCRIPT"
    except Exception:
        pass

    # 2) Ultralytics
    try:
        info = try_ultralytics(path)
        print("%s✓ 로드 성공%s — %s,  클래스 %d개"
              % (G, X, info["type"], info["nc"]))
        if info.get("forward"):
            print("    %s추론 1회 통과%s" % (G, X))
        else:
            print("    %s추론 실패:%s %s" % (R, X, info.get("forward_err")))

        print("\n  클래스 목록:")
        for i, n in sorted(info["names"].items()):
            print("    %2d  %s" % (i, n))

        cmp_out = compare_legacy(stem, info["names"])
        if cmp_out:
            print("\n" + cmp_out)
        return "ULTRALYTICS"

    except Exception as e:
        print("%s✗ 로드 실패%s" % (R, X))
        msg = str(e)
        print("    %s" % msg[:400])

        low = msg.lower()
        print()
        if "can't get attribute" in low or "no module named 'ultralytics" in low:
            print("  %s→ ultralytics 버전 불일치입니다.%s" % (Y, X))
            print("     학습에 쓴 버전을 작년 팀에 확인하고 그 버전으로 설치하세요.")
        elif "modulenotfounderror" in low:
            print("  %s→ 가중치가 참조하는 모듈이 없습니다.%s" % (Y, X))
            print("     YOLOv12 저자 fork로 학습됐을 가능성이 높습니다.")
            print("     메인라인 ultralytics로는 열 수 없습니다.")
        else:
            print("  %s→ 전체 트레이스백:%s" % (Y, X))
            traceback.print_exc()
        return "FAIL"


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    try:
        import torch
    except ImportError:
        sys.exit("torch 가 필요합니다:  pip install torch")

    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += sorted(glob.glob(os.path.join(a, "*.pt")))
        elif os.path.isfile(a):
            paths.append(a)
        else:
            print("%s경로 없음: %s%s" % (Y, a, X))

    if not paths:
        sys.exit("검사할 .pt 파일을 찾지 못했습니다.")

    print("torch %s / python %s" % (torch.__version__, sys.version.split()[0]))
    try:
        import ultralytics
        print("ultralytics %s" % ultralytics.__version__)
    except ImportError:
        print("%sultralytics 미설치 — YOLOv12 계열은 검사 못 합니다%s" % (Y, X))

    results = {}
    for p in paths:
        results[p] = check_one(p, torch)

    print("\n" + "=" * 72)
    print("요약")
    print("=" * 72)
    for p, r in results.items():
        mark = {"TORCHSCRIPT": G + "OK  " + X, "ULTRALYTICS": G + "OK  " + X,
                "LFS_POINTER": R + "LFS " + X, "FAIL": R + "FAIL" + X}[r]
        print("  %s  %s" % (mark, os.path.basename(p)))
    print()


if __name__ == "__main__":
    main()
