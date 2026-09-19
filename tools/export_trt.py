#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_trt.py — classes.yaml 을 단일 기준으로 .pt -> .engine 변환

  python3 tools/export_trt.py --section obstacle
  python3 tools/export_trt.py --section traffic_light --half
  python3 tools/export_trt.py --all

★ 반드시 차량 노트북에서 실행하십시오.
  .engine 은 GPU 아키텍처·드라이버·TensorRT 버전에 묶입니다. 랩실 PC 에서 만든
  파일은 차량에서 열리지 않거나, 더 나쁘게는 열리고 결과가 다릅니다.

★ imgsz 를 인자로 받지 않습니다.
  학습 = export = 런타임 세 군데가 어긋나면 에러 없이 작은 객체만 조용히
  사라집니다. 유일한 기준은 classes.yaml 이고, 여기서 그것만 읽습니다.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

# class_map 은 rospy/rospkg 에 의존합니다. 이 도구는 ROS 마스터 없이도
# 돌아야 하므로(변환 중에는 어차피 모든 노드를 내려놓습니다), 같은 규칙을
# 재구현하는 대신 classes.yaml 을 직접 읽습니다.
# ★ 정규화 규칙은 utils/class_map.py 의 _norm 과 동일하게 유지할 것.
import yaml  # noqa: E402

_PREFIX = re.compile(r"^\d+_")


def _norm(name):
    return _PREFIX.sub("", str(name)).strip()


def load_section(section):
    path = os.path.join(PKG, "config", "classes.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if section not in cfg:
        print("classes.yaml 에 '%s' 섹션이 없습니다" % section)
        sys.exit(1)
    sec = cfg[section]
    sec["names"] = {int(k): _norm(v) for k, v in sec.get("names", {}).items()}
    return sec


def verify_names(model_names, expected, section):
    got = {int(k): _norm(v) for k, v in dict(model_names).items()}
    if got == expected:
        return
    print("\n★ [%s] 클래스 불일치 — 변환 중단" % section)
    for i in sorted(set(got) | set(expected)):
        a, b = expected.get(i, "(없음)"), got.get(i, "(없음)")
        print("  %d: classes.yaml=%-14s 모델=%-14s %s"
              % (i, a, b, "" if a == b else "  <-- 다름"))
    print("\n  이름만 같고 순서가 다른 경우가 가장 위험합니다.")
    print("  좌우 지시가 정반대로 학습된 모델일 수 있습니다.")
    sys.exit(1)

# YOLOPv2 는 ultralytics 모델이 아니라 순수 TorchScript 라
# YOLO().export() 경로를 탈 수 없습니다. 아래 주석 참고.
UNSUPPORTED = {"lane"}


def preflight(min_free_mb):
    """다른 프로세스가 GPU 를 점유한 채 export 하면 변환 도중 OOM 으로 죽고,
    실패한 .engine 이 남아 다음 실행에서 '있는데 안 열리는' 상태가 됩니다."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            encoding="utf-8").strip().split(", ")
        total, used, driver = int(out[0]), int(out[1]), out[2]
    except Exception as e:
        print("nvidia-smi 실패: %s" % e)
        print("GPU 상태를 확인할 수 없습니다. 계속하려면 --force")
        return None
    free = total - used
    print("GPU  total %d MB / used %d MB / free %d MB  driver %s"
          % (total, used, free, driver))
    if free < min_free_mb:
        print("\n★ 여유 VRAM 이 %d MB 미만입니다." % min_free_mb)
        print("  perception / 판단·제어 노드를 모두 내리고 다시 실행하십시오.")
        print("  점유 프로세스:")
        subprocess.call(["nvidia-smi",
                         "--query-compute-apps=pid,process_name,used_memory",
                         "--format=csv"])
        sys.exit(1)
    return driver


def export_one(section, half, workspace, driver, force):
    from ultralytics import YOLO

    if section in UNSUPPORTED:
        print("\n[%s] 건너뜀 — ultralytics 모델이 아닙니다 (YOLOPv2 TorchScript)."
              % section)
        print("  변환하려면 ONNX -> trtexec 경로가 따로 필요합니다. 파일 하단 주석 참고.")
        return None

    sec = load_section(section)
    pt = os.path.join(PKG, sec["file"])
    imgsz = int(sec["imgsz"])
    expected = sec["names"]

    if not os.path.exists(pt):
        print("\n[%s] 가중치 없음: %s" % (section, pt))
        return None

    print("\n" + "=" * 60)
    print("[%s] %s  imgsz=%d  half=%s" % (section, sec["file"], imgsz, half))

    model = YOLO(pt)

    # 엉뚱한 .pt 를 변환하면 런타임에서야 class_map 이 잡아냅니다.
    # 여기서 먼저 막는 편이 몇 분 빠릅니다.
    verify_names(model.names, expected, section)
    print("클래스 대조 통과: %s" % list(model.names.values()))

    out = os.path.splitext(pt)[0] + ".engine"
    if os.path.exists(out) and not force:
        print("이미 존재: %s  (덮어쓰려면 --force)" % out)
        return None

    t0 = time.time()
    # dynamic=False, batch=1 고정: 런타임이 항상 단일 프레임이고,
    # 동적 shape 는 최적화 여지를 줄이는 대신 얻는 게 없습니다.
    path = model.export(format="engine", imgsz=imgsz, half=half,
                        device=0, batch=1, dynamic=False,
                        workspace=workspace, verbose=False)
    dt = time.time() - t0

    meta = {
        "section": section,
        "source_pt": sec["file"],
        "imgsz": imgsz,
        "half": bool(half),
        "driver": driver,
        "names": {int(k): v for k, v in model.names.items()},
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": os.uname().nodename,
    }
    with open(os.path.splitext(pt)[0] + ".engine.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("완료 %.0f초 -> %s" % (dt, path))
    print("  ↑ 이 .engine 은 %s 의 드라이버 %s 전용입니다. git 에 넣지 마십시오."
          % (meta["hostname"], driver))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", nargs="+",
                    help="classes.yaml 의 섹션명 (traffic_light, obstacle, cone)")
    ap.add_argument("--all", action="store_true",
                    help="변환 가능한 섹션 전부")
    ap.add_argument("--half", action="store_true",
                    help="FP16. VRAM 절반, 속도 향상. 단 작은 객체 손실 위험 — 아래 참고")
    ap.add_argument("--workspace", type=int, default=4,
                    help="TensorRT 빌드 워크스페이스 GB (기본 4)")
    ap.add_argument("--min-free", type=int, default=5000,
                    help="필요한 최소 여유 VRAM MB (기본 5000)")
    ap.add_argument("--force", action="store_true",
                    help="기존 .engine 덮어쓰기")
    a = ap.parse_args()

    if not a.section and not a.all:
        ap.error("--section 또는 --all 중 하나가 필요합니다")

    sections = a.section or ["traffic_light", "obstacle"]

    driver = preflight(a.min_free)
    if driver is None and not a.force:
        sys.exit(1)

    if a.half:
        print("\n★ FP16 으로 변환합니다.")
        print("  obstacle 의 LED 패널은 imgsz 960 에서 약 25 px 입니다.")
        print("  FP16 이 이 크기의 객체를 떨어뜨리는 사례가 있으니,")
        print("  변환 후 bench.py 의 검출 대조를 반드시 확인하십시오.")

    done = []
    for s in sections:
        try:
            p = export_one(s, a.half, a.workspace, driver, a.force)
            if p:
                done.append((s, p))
        except SystemExit:
            raise
        except Exception as e:
            print("\n[%s] 실패: %s: %s" % (s, type(e).__name__, e))

    if not done:
        return

    print("\n" + "=" * 60)
    print("다음 — 속도와 '검출이 같은지' 를 둘 다 확인하십시오.")
    print("속도만 보고 넘어가면 조용히 누락된 객체를 놓칩니다.\n")
    for s, p in done:
        sec = load_section(s)
        rel = os.path.relpath(p, PKG)
        print("  python3 tools/bench.py \\")
        print("      --model %s %s \\" % (sec["file"], rel))
        print("      --src videos/driving.mp4 --imgsz %d --n 200\n" % sec["imgsz"])
    print("주기 배분은 평균이 아니라 p95 를 기준으로 잡으십시오.")


# ─────────────────────────────────────────────────────────────────────
# lane (YOLOPv2) 를 변환하려면
#
#   ultralytics 를 못 쓰므로 두 단계로 나뉩니다.
#
#     1) TorchScript -> ONNX
#        torch.jit.load 로 올린 뒤 torch.onnx.export.
#        입력 shape 는 utils.py 의 letterbox 결과와 일치해야 합니다
#        (lane.launch 의 imgsz 확인). opset 12 이상.
#
#     2) ONNX -> engine
#        trtexec --onnx=lane.onnx --saveEngine=lane.engine --fp16
#
#   그리고 lane_detector.py 의 torch.jit.load 부분을 TRT 런타임으로
#   바꿔야 합니다 — 노드 코드 수정이 따릅니다.
#
#   ★ 9/20 까지 남은 일정에서 이 작업을 권하지 않습니다.
#     lane.pt 는 아직 차량에서 검증조차 안 된 상태입니다. 쓸 수 있는지
#     모르는 모델을 최적화하는 것은 순서가 뒤바뀐 일입니다.
#     VRAM 이 급하면 lane 을 내리는 쪽이 빠르고 되돌리기도 쉽습니다.
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
