#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stopline.py — YOLOPv2 차선 마스크에서 '정지선'만 기하학적으로 분리합니다.

  배치 위치:  mando_vision_2026/utils/stopline.py

────────────────────────────────────────────────────────────────────────
왜 이게 필요한가

YOLOPv2 의 lane_line 헤드는 2채널 이진 세그멘테이션입니다.
차선 / 정지선 / 화살표 / 횡단보도를 구분하는 클래스 인덱스가 없습니다.
전부 "노면 마킹"으로 한 덩어리로 나옵니다.

그런데 이미지 상에서 방향이 다릅니다.
  차선, 정면 접근 중인 횡단보도 줄무늬 → 소실점으로 수렴, 세로로 김
  정지선                               → 진행방향에 수직, 가로로 김

연결요소별 주축 방향(PCA)으로 거르면 정지선만 남습니다.
lane 추론은 이미 돌고 있으므로 GPU 비용은 0, CPU 후처리 1~2 ms 입니다.

────────────────────────────────────────────────────────────────────────
한계 — 반드시 알고 쓰세요

▣ '흰 가로줄'이면 무엇이든 잡습니다.
  횡단보도를 비스듬히 볼 때, 노면 문자, 도로 이음매, 강한 그림자 경계도
  조건에 맞으면 잡힙니다. 반드시 자기 영상으로 임계값을 튜닝하세요.

▣ D455 는 신호등을 보려고 위를 향합니다.
  그래서 정지선이 가까워지면 화면 아래로 빠져나가 사라집니다.
  즉 이 모듈은 "정지선이 저 앞 N m 에 있다"는 조기 감지용이고,
  최종 정밀 정지는 아래를 보는 C920 이 담당해야 합니다. 역할이 다릅니다.

▣ 거리 환산은 평평한 지면을 가정합니다.
  경사로에서는 오차가 커집니다. 오르막이면 실제보다 멀게 나옵니다.

▣ 기울기 허용치는 min_run_ratio 가 결정합니다.
  한 행이 정지선을 가로지르는 길이는 대략 (선 두께 / sin(기울기)) 입니다.
  기본값 0.35 → 약 ±7도까지. 0.15 로 낮추면 ±15도까지 잡히지만
  가로 그림자·노면 이음매 오검출이 늘어납니다.
  차량이 차로에 정렬된 채 정지선에 접근한다면 기본값으로 충분합니다.

▣ 합성 마스크 12케이스 검증 결과 (2026-09-09)
  차선 접촉 / 횡단보도 인접(3px) / 점선형 / 두께 10px  → 모두 검출
  차선만 / 횡단보도만                                → 오검출 없음
  가로 그림자 경계                                   → 오검출 발생 ★
  마지막 항목은 기하로 못 거릅니다. 네트워크가 이미 '차선'이라고
  칠한 것이라, 시간 투표(StopLineVoter)와 거리 게이트로 막아야 합니다.
  후처리 지연 3.7 ms/frame (720x1280, i7급 CPU 1코어)

────────────────────────────────────────────────────────────────────────
좌표계 메모

lane_line_mask() 출력은 720x1280 이고, 이는 입력을 1280x720 으로
리사이즈한 이미지와 픽셀 1:1 대응입니다.
(384x640 letterbox 에서 위아래 패딩 12행을 crop 12:372 로 제거 후 x2 업샘플)

D455 를 color_width:=1280 color_height:=720 으로 띄우면
camera_info 의 K 를 스케일 보정 없이 그대로 쓸 수 있습니다.

────────────────────────────────────────────────────────────────────────
단독 실행 (임계값 튜닝용)

  python3 utils/stopline.py --model weights/lane.pt --src videos/driving.mp4
  python3 utils/stopline.py --model weights/lane.pt --src videos/stopline.mp4 \
      --fy 640 --cy 360 --cam-h 0.67 --pitch 0

  트랙바로 조건을 조절하고, s 로 현재 설정을 화면에 출력합니다.
"""

import math

import numpy as np
import cv2


# ═══════════════════════════════════════════════════════════════════════
#  파라미터
# ═══════════════════════════════════════════════════════════════════════
class StopLineParams(object):
    """튜닝 후 이 값들을 launch 파일 rosparam 으로 빼세요."""

    def __init__(self):
        # ── ROI (마스크 크기에 대한 비율) ─────────────────────────────
        self.roi_top = 0.45      # 이 위쪽은 무시 (지평선/원거리)
        self.roi_x0 = 0.12      # 좌측 잘라내기
        self.roi_x1 = 0.88      # 우측 잘라내기

        # ── 행 단위 런-길이 조건 (핵심) ──────────────────────────────
        # 정지선  : 그 행에 '아주 긴 연속 구간'이 1개
        # 차선    : 짧은 구간 2개
        # 횡단보도: 짧은 구간 여러 개  ← 구간 개수로 걸러집니다
        self.min_run_ratio = 0.35   # ROI 폭 대비 최소 연속 길이
        self.max_seg = 2      # 그 행의 최대 구간 개수
        self.max_band_h = 70     # 정지선 밴드 최대 두께(px). 넘으면 다른 것
        self.min_band_h = 3      # 최소 두께 (노이즈 제거)

        # ── 전처리 ───────────────────────────────────────────────────
        # close_h 는 1 로 두세요. 세로로 이으면 횡단보도·차선과 붙습니다.
        self.open_w = 5      # 가로 opening (점 노이즈 제거)
        self.close_w = 25     # 가로 close (점선/끊긴 정지선 잇기)
        self.close_h = 1

        # ── 시간 필터 ────────────────────────────────────────────────
        self.vote_n = 5        # 최근 N 프레임 중
        self.vote_k = 3        # K 프레임 이상에서 검출되어야 확정


def _orientation_deg(xs, ys):
    """연결요소의 주축 방향. 0 도 = 수평."""
    xm = xs.mean()
    ym = ys.mean()
    dx = xs - xm
    dy = ys - ym
    cxx = float((dx * dx).mean())
    cyy = float((dy * dy).mean())
    cxy = float((dx * dy).mean())
    cov = np.array([[cxx, cxy], [cxy, cyy]])
    vals, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, int(np.argmax(vals))]
    ang = math.degrees(math.atan2(vy, vx))
    if ang > 90.0:
        ang -= 180.0
    if ang < -90.0:
        ang += 180.0
    return ang


# ═══════════════════════════════════════════════════════════════════════
#  핵심 — 마스크에서 정지선 후보 찾기
# ═══════════════════════════════════════════════════════════════════════
def find_stopline(ll_mask, p):
    """
    ll_mask : lane_line_mask() 출력 (H,W) 0/1
    p       : StopLineParams

    반환 None 또는 dict:
        v_near : 정지선 아래쪽 끝의 픽셀 행 (가장 가까운 지점)
        v_mid  : 중심 행
        box    : (x, y, w, h)  밴드 외접사각형
        angle  : 수평 대비 각도(deg)
        run    : 밴드 내 최대 연속 길이(px)
        n_row  : 밴드 두께(행 수)

    ── 방식 ──
    연결요소로 뽑으면 정지선이 차선·횡단보도와 맞닿는 순간 하나로 뭉쳐
    전부 탈락합니다(실측으로 확인됨). 그래서 '행 단위'로 봅니다.

      정지선인 행   : 아주 긴 연속 구간이 1개
      차선인 행     : 짧은 구간 2개
      횡단보도인 행 : 짧은 구간 여러 개

    붙어 있어도 행마다 독립적으로 판정되므로 영향받지 않습니다.
    """
    m = (np.asarray(ll_mask) > 0).astype(np.uint8)
    h, w = m.shape[:2]

    y0 = int(h * p.roi_top)
    x0 = int(w * p.roi_x0)
    x1 = int(w * p.roi_x1)
    roi_w = max(x1 - x0, 1)

    # ROI 부분배열만 다룹니다 (전체 프레임 연산은 지연이 2~3배)
    roi = m[y0:, x0:x1].copy()
    hr = roi.shape[0]

    if p.open_w > 1:      # 점 노이즈 제거
        roi = cv2.morphologyEx(
            roi, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (int(p.open_w), 1)))
    if p.close_w > 1:     # 점선/끊긴 정지선 잇기 (세로로는 잇지 않음)
        roi = cv2.morphologyEx(
            roi, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (int(p.close_w), max(1, int(p.close_h)))))

    # ── 행별 연속 구간 길이 / 개수 (완전 벡터화) ─────────────────────
    b = (roi > 0).astype(np.int8)
    pad = np.zeros((hr, 1), np.int8)
    d = np.diff(np.hstack([pad, b, pad]), axis=1)
    sr, sc = np.nonzero(d == 1)     # 구간 시작 (행, 열)
    er, ec = np.nonzero(d == -1)    # 구간 끝
    if sr.size == 0:
        return None
    lengths = (ec - sc).astype(np.int32)

    max_run = np.zeros(hr, np.int32)
    np.maximum.at(max_run, sr, lengths)
    n_seg = np.bincount(sr, minlength=hr)

    min_run = int(p.min_run_ratio * roi_w)
    ok = (max_run >= min_run) & (n_seg <= p.max_seg) & (n_seg > 0)
    if not ok.any():
        return None

    # ── 연속한 ok 행들을 밴드로 묶고, 가장 아래(가까운) 밴드 선택 ────
    idx = np.nonzero(ok)[0]
    splits = np.nonzero(np.diff(idx) > 1)[0]
    bands = np.split(idx, splits + 1)

    best = None
    for band in bands:
        nrow = band.size
        if nrow < p.min_band_h or nrow > p.max_band_h:
            continue
        if best is None or band[-1] > best[-1]:
            best = band
    if best is None:
        return None

    ys = best

    # 각 행의 '가장 긴 구간'만 모아 밴드 픽셀을 구성합니다.
    # (같은 행의 짧은 차선 구간이 섞이면 각도가 오염됩니다)
    px = []
    py = []
    x_lo, x_hi = 10 ** 9, -10 ** 9
    for y in ys:
        sel_y = np.nonzero(sr == y)[0]
        if sel_y.size == 0:
            continue
        j = sel_y[int(np.argmax(lengths[sel_y]))]
        s, e = int(sc[j]), int(ec[j])
        x_lo = min(x_lo, s)
        x_hi = max(x_hi, e)
        step = max(1, (e - s) // 40)          # 각도 계산용 서브샘플
        xs_row = np.arange(s, e, step, dtype=np.float64)
        px.append(xs_row)
        py.append(np.full(xs_row.size, float(y)))

    if not px:
        return None
    px = np.concatenate(px)
    py = np.concatenate(py)
    ang = _orientation_deg(px, py) if px.size >= 8 else 0.0

    by = int(ys[0]) + y0
    bh = int(ys[-1] - ys[0] + 1)
    return dict(v_near=int(ys[-1]) + y0, v_mid=int(ys[len(ys) // 2]) + y0,
                box=(int(x_lo) + x0, by, int(x_hi - x_lo), bh),
                angle=float(ang), run=int(max_run[ys].max()), n_row=int(bh))


# ═══════════════════════════════════════════════════════════════════════
#  픽셀 행 → 지면 거리 (평지 가정)
# ═══════════════════════════════════════════════════════════════════════
def row_to_distance(v, fy, cy, cam_height, pitch_deg):
    """
    v          : 픽셀 행 (1280x720 이미지 기준)
    fy, cy     : camera_info K[4], K[5]
    cam_height : 지면에서 카메라 광학중심까지 (m)  ← URDF front_z
    pitch_deg  : 아래를 보면 +, 위를 보면 −        ← URDF front_pitch

    반환 : 카메라로부터의 지면 수평거리 (m). 지평선 위/무한대면 None
    """
    alpha = math.atan2(float(v) - float(cy), float(fy))   # 광축 아래 각
    phi = math.radians(pitch_deg) + alpha                 # 수평 기준 내려본 각
    if phi <= 1e-3:
        return None
    d = cam_height / math.tan(phi)
    if not np.isfinite(d) or d <= 0 or d > 100.0:
        return None
    return d


def to_base_link(d_cam, cam_x):
    """카메라 기준 거리 → base_link(뒷차축) 기준 거리. cam_x = URDF front_x"""
    return None if d_cam is None else d_cam + float(cam_x)


# ═══════════════════════════════════════════════════════════════════════
#  시간 필터 — 단발 오검출 제거
# ═══════════════════════════════════════════════════════════════════════
class StopLineVoter(object):
    def __init__(self, p):
        self.p = p
        self.buf = []

    def update(self, det):
        self.buf.append(det)
        if len(self.buf) > self.p.vote_n:
            self.buf.pop(0)
        hits = [d for d in self.buf if d is not None]
        if len(hits) < self.p.vote_k:
            return None
        # 확정 시 거리는 중앙값 (튀는 프레임 억제)
        vs = sorted(d["v_near"] for d in hits)
        med = vs[len(vs) // 2]
        out = dict(hits[-1])
        out["v_near"] = med
        out["n_hit"] = len(hits)
        return out


# ═══════════════════════════════════════════════════════════════════════
#  단독 실행 — 임계값 튜닝
# ═══════════════════════════════════════════════════════════════════════
def _tune():
    import argparse
    import os
    import sys
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="weights/lane.pt")
    ap.add_argument("--src", required=True, help="영상파일 / /dev/xxx / 토픽")
    ap.add_argument("--utils", default=None)
    # 거리 환산용 (D455 영상일 때만 의미 있음)
    ap.add_argument("--fy", type=float, default=None, help="camera_info K[4]")
    ap.add_argument("--cy", type=float, default=None, help="camera_info K[5]")
    ap.add_argument("--cam-h", dest="cam_h", type=float, default=0.67,
                    help="지면~카메라 높이 m (URDF front_z)")
    ap.add_argument("--pitch", type=float, default=0.0,
                    help="아래 +, 위 − (URDF front_pitch)")
    ap.add_argument("--cam-x", dest="cam_x", type=float, default=0.75)
    a = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    utils_dir = a.utils or os.path.join(
        os.path.dirname(os.path.abspath(a.model)), "..", "utils")
    utils_dir = os.path.abspath(utils_dir)
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)

    net = torch.jit.load(a.model, map_location=dev).eval()
    half = (dev == "cuda")
    if half:
        net.half()
    from utils import letterbox, lane_line_mask
    print("lane.pt 로드 완료  device=%s" % dev)

    cap = cv2.VideoCapture(a.src if not a.src.startswith("/dev/") else a.src)
    if not cap.isOpened():
        sys.exit("입력을 열 수 없습니다: %s" % a.src)

    p = StopLineParams()
    voter = StopLineVoter(p)

    WIN = "stopline tune   space:정지  s:설정출력  q:종료"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 760)
    # ★ 트랙바는 현재 StopLineParams 의 필드와 1:1 로 맞춰야 합니다.
    #   (연결요소 방식에서 행-런 방식으로 바꾸면서 필드 이름이 전부 바뀌었습니다)
    cv2.createTrackbar("roi_top %", WIN, int(p.roi_top * 100), 90, lambda v: None)
    cv2.createTrackbar("min_run %", WIN, int(p.min_run_ratio * 100), 90, lambda v: None)
    cv2.createTrackbar("max_seg", WIN, int(p.max_seg), 8, lambda v: None)
    cv2.createTrackbar("max_band_h", WIN, int(p.max_band_h), 200, lambda v: None)
    cv2.createTrackbar("min_band_h", WIN, int(p.min_band_h), 40, lambda v: None)
    cv2.createTrackbar("open_w", WIN, p.open_w, 81, lambda v: None)
    cv2.createTrackbar("close_w", WIN, p.close_w, 61, lambda v: None)

    paused = False
    frame = None
    while True:
        if not paused:
            ok, bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = bgr

        if frame is None:
            break

        p.roi_top = max(cv2.getTrackbarPos("roi_top %", WIN), 0) / 100.0
        p.min_run_ratio = max(cv2.getTrackbarPos("min_run %", WIN), 1) / 100.0
        p.max_seg = max(cv2.getTrackbarPos("max_seg", WIN), 1)
        p.max_band_h = max(cv2.getTrackbarPos("max_band_h", WIN), 2)
        p.min_band_h = max(cv2.getTrackbarPos("min_band_h", WIN), 1)
        p.open_w = max(cv2.getTrackbarPos("open_w", WIN), 1)
        p.close_w = max(cv2.getTrackbarPos("close_w", WIN), 1)

        im = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img, _, _ = letterbox(im, 640, stride=32)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        t = torch.from_numpy(x).to(dev)
        t = (t.half() if half else t.float()) / 255.0

        t0 = time.time()
        with torch.no_grad():
            _, _, ll = net(t.unsqueeze(0))
        lm = lane_line_mask(ll)
        ms_net = (time.time() - t0) * 1000.0

        t1 = time.time()
        raw = find_stopline(lm, p)
        det = voter.update(raw)
        ms_post = (time.time() - t1) * 1000.0

        vis = im.copy()
        vis[lm > 0] = (0, 0, 255)

        # ROI 표시
        y0 = int(lm.shape[0] * p.roi_top)
        cv2.line(vis, (0, y0), (vis.shape[1], y0), (255, 255, 0), 1)

        txt2 = ""
        if raw is not None:
            bx, by, bw, bh = raw["box"]
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 200, 255), 2)
        if det is not None:
            v = det["v_near"]
            cv2.line(vis, (0, v), (vis.shape[1], v), (0, 255, 0), 3)
            d = None
            if a.fy and a.cy:
                d = row_to_distance(v, a.fy, a.cy, a.cam_h, a.pitch)
                d = to_base_link(d, a.cam_x)
            txt2 = "STOPLINE  v=%d  ang=%+.1f  run=%dpx  두께=%dpx  %s" % (
                v, det["angle"], det["run"], det["n_row"],
                ("dist=%.2f m" % d) if d else "dist=? (--fy --cy 필요)")
        else:
            txt2 = "정지선 없음"

        cv2.rectangle(vis, (0, 0), (vis.shape[1], 52), (0, 0, 0), -1)
        cv2.putText(vis, "net %.0f ms   post %.1f ms" % (ms_net, ms_post),
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(vis, txt2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if det else (120, 120, 120), 2, cv2.LINE_AA)
        cv2.imshow(WIN, vis)

        k = cv2.waitKey(1 if not paused else 30) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord(' '):
            paused = not paused
        elif k == ord('s'):
            print("\n── 현재 설정 (launch 에 넣으세요) ──")
            for key in ("roi_top", "roi_x0", "roi_x1", "min_run_ratio",
                        "max_seg", "max_band_h", "min_band_h",
                        "open_w", "close_w", "close_h", "vote_n", "vote_k"):
                print('  <param name="stopline/%s" value="%s"/>'
                      % (key, getattr(p, key)))
            print()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    _tune()
