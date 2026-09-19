#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obstacle_depth.py — bbox + D455 정렬 뎁스 -> base_link 점. 지면 투영과 대조해 맞을 때만 뎁스를 쓴다. ROS 비의존.

  사용처: src/obstacle_depth.py (노드),  tools/verify_obstacle_depth.py,  tools/bag_obstacle_depth.py

★ 왜 뎁스를 그대로 믿지 않는가
  D455 스테레오 뎁스는 역광·반사·저질감 면에서 구멍이 나거나 엉뚱한 거리를 낸다.
  지면 투영(utils/ground.py)은 조명과 무관하지만 밑동이 잘리면(≤2.70 m) 쓸 수 없고
  멀수록 거리 해상도가 무너진다. 서로 약한 곳이 달라서, 둘이 맞으면 뎁스(가까운 면을
  직접 잰 값)를 쓰고, 안 맞거나 뎁스가 부족하면 지면 투영으로 떨어진다.
  어느 쪽을 썼는지(used)와 이유(reason)를 항목마다 남겨 BEV 에서 눈으로 볼 수 있게 한다.

★ 카메라 기하는 ground.py 의 Ground 한 벌만 쓴다 (K, 높이, pitch, 장착 위치).
  뎁스 점과 지면 투영이 같은 외부파라미터를 쓰므로, 둘의 차이는 순수하게
  '뎁스 거리 vs 평지 가정 거리' 차이다. URDF 와의 일치는 노드가 시동 때 TF 로 확인한다.

used / reason
  depth  ok        뎁스와 지면 투영이 허용오차 안에서 맞음
  depth  cut_ok    밑동이 잘려 지면 투영이 2.70 m 로 뭉개짐 — 뎁스가 그보다 가까우면 뎁스를 믿는다
  ground far       지면 투영 거리가 depth_max_m 밖 (D455 오차가 거리 제곱으로 커지는 구간)
  ground no_depth  이 프레임과 맞는 뎁스 영상이 없음
  ground few_valid 유효 뎁스 열이 min_cols 미만 (역광·반사 구멍)
  ground mismatch  뎁스와 지면 투영이 허용오차 밖 (뎁스 오인식 의심)
  ground cut_far   밑동이 잘렸는데 뎁스가 2.70 m 보다 멀다 — 둘 중 무엇도 못 믿어 지면 값(더 가까움)을 쓴다
  -      no_ground 밑변이 수평선 위라 지면 투영 불가 — 대조할 수 없어 점을 내지 않는다
"""
import math

import numpy as np

from obstacle_ground import Projector


class DepthProjector(object):

    def __init__(self, G, wheel_radius=0.150, cols=5, col_frac=0.6,
                 band=(0.5, 0.95), percentile=30.0, min_valid=0.2, min_cols=3,
                 depth_min_m=0.3, depth_max_m=8.0, tol_abs_m=0.30, tol_rel=0.15,
                 pitch_tol_deg=0.5, image_width=1280, image_height=720):
        self.G = G
        self.proj = Projector(G, pitch_tol_deg)
        self.wheel_radius = float(wheel_radius)
        self.cols = int(cols)
        self.col_frac = float(col_frac)
        self.band = (float(band[0]), float(band[1]))
        # 가까운 면을 재려는 것. bbox 안에는 물체 뒤의 지면·배경이 섞여 중앙값은 멀리 끌린다.
        self.percentile = float(percentile)
        self.min_valid = float(min_valid)
        self.min_cols = int(min_cols)
        self.depth_min = float(depth_min_m)
        self.depth_max = float(depth_max_m)
        self.tol_abs = float(tol_abs_m)
        self.tol_rel = float(tol_rel)
        self.W = int(image_width)
        self.H = int(image_height)
        self._c, self._s = math.cos(G.th), math.sin(G.th)

    # ── 픽셀 + 뎁스 -> base_link ─────────────────────────────────
    def backproject(self, u, v, Z):
        """배열 u, v [px], Z [m] -> base_link (x, y, z). Ground.to_ground 와 같은 회전."""
        G = self.G
        p = (v - G.cy) / G.fy
        q = (u - G.cx) / G.fx
        fwd = Z * (self._c - p * self._s)
        down = Z * (p * self._c + self._s)
        return G.x_off + fwd, G.y_off - q * Z, G.h - down - self.wheel_radius

    def ground_points(self, box):
        """bbox 밑변을 cols 점으로 나눠 지면 투영. 뎁스를 못 쓸 때 내보내는 점."""
        x1, _y1, x2, y2 = box
        out = []
        for k in range(self.cols):
            u = x1 + (x2 - x1) * k / max(self.cols - 1, 1)
            g = self.G.to_ground(u, y2)
            if g is not None:
                out.append((g[0], g[1], -self.wheel_radius))
        return out

    def depth_columns(self, box, depth_m):
        """bbox 를 가로 cols 칸으로 나눠 칸마다 가까운 면 한 점. 유효 픽셀이 모자란 칸은 None."""
        dh, dw = depth_m.shape[:2]
        sx, sy = dw / float(self.W), dh / float(self.H)
        x1, y1, x2, y2 = [float(v) for v in box]
        w, h = x2 - x1, y2 - y1
        r0 = int(max(0, math.floor((y1 + h * self.band[0]) * sy)))
        r1 = int(min(dh, math.ceil((y1 + h * self.band[1]) * sy)))
        out = []
        for k in range(self.cols):
            uc = x1 + w * (k + 0.5) / self.cols
            half = w / self.cols * self.col_frac * 0.5
            c0 = int(max(0, math.floor((uc - half) * sx)))
            c1 = int(min(dw, math.ceil((uc + half) * sx)))
            if r1 <= r0 or c1 <= c0:
                out.append(None)
                continue
            patch = depth_m[r0:r1, c0:c1]
            ok = np.isfinite(patch) & (patch >= self.depth_min) & (patch <= self.depth_max * 1.5)
            n_ok = int(ok.sum())
            if n_ok == 0 or n_ok < self.min_valid * patch.size:
                out.append(None)
                continue
            rr, cc = np.nonzero(ok)
            u = (cc + c0 + 0.5) / sx
            v = (rr + r0 + 0.5) / sy
            X, Y, Zb = self.backproject(u, v, patch[ok])
            out.append({"x": float(np.percentile(X, self.percentile)),
                        "y": float(np.median(Y)), "z": float(np.median(Zb)),
                        "valid": round(n_ok / float(patch.size), 2)})
        return out

    # ── 한 박스 판정 ─────────────────────────────────────────────
    def measure(self, box, depth_m=None):
        x1, y1, x2, y2 = [float(v) for v in box]
        cut = y2 >= self.H - 2
        g = self.proj.project(x1, y1, x2, y2)
        res = {"box": [int(x1), int(y1), int(x2), int(y2)], "cut": cut,
               "ground": None, "depth": None, "used": None, "reason": None, "points": []}
        if g is None:
            res["reason"] = "no_ground"
            return res
        res["ground"] = {"x": round(g["x"], 3), "y": round(g["y"], 3),
                         "x_err": None if g["x_err"] is None else round(g["x_err"], 3)}
        gpts = self.ground_points((x1, y1, x2, y2))

        def fallback(reason):
            res.update(used="ground", reason=reason, points=gpts)
            return res

        if not cut and g["x"] > self.depth_max:
            return fallback("far")
        if depth_m is None:
            return fallback("no_depth")
        cols = [c for c in self.depth_columns((x1, y1, x2, y2), depth_m) if c is not None]
        if len(cols) < self.min_cols:
            res["depth"] = {"cols": len(cols)}
            return fallback("few_valid")
        dx = float(np.median([c["x"] for c in cols]))
        dy = float(np.median([c["y"] for c in cols]))
        res["depth"] = {"x": round(dx, 3), "y": round(dy, 3), "cols": len(cols)}

        if cut:
            # 지면 투영 x 는 '이보다 가깝다' 는 상한일 뿐이다
            if dx > g["x"] + self.tol_abs:
                return fallback("cut_far")
            reason = "cut_ok"
        else:
            tol = max(self.tol_abs, self.tol_rel * g["x"], g["x_err"] or 0.0)
            res["depth"]["tol"] = round(tol, 3)
            if abs(dx - g["x"]) > tol:
                return fallback("mismatch")
            reason = "ok"
        res.update(used="depth", reason=reason,
                   points=[(c["x"], c["y"], c["z"]) for c in cols])
        return res


def fit_ground_plane(depth_m, fx, fy, cx, cy, rows=(480, 720), cols=(320, 960), stride=2,
                     depth_range=(0.5, 6.0), inlier_m=0.03, iters=200, min_points=2000, seed=0):
    """차 앞 노면 뎁스로 평면을 맞춰 카메라 높이·pitch·roll 을 잰다. 실패하면 None.

    ★ 2026-09-16 송도 bag 에서 이것으로 pitch −2.75° (당시 URDF −1.0°, 09-17 반영) 를 찾았다. 지면 투영 거리는
      pitch 1.75° 틀리면 5 m 에서 0.4 m, 7 m 에서 0.9 m 짧아져
      대조 허용오차를 거의 다 잡아먹는다. 그 위에 bbox 여유만 조금 얹혀도 멀쩡한 뎁스가 mismatch 로 떨어진다.
      반환 pitch_deg 는 ground.py 규약 (아래를 보면 +).
    """
    d = depth_m[rows[0]:rows[1]:stride, cols[0]:cols[1]:stride]
    v, u = np.mgrid[rows[0]:rows[1]:stride, cols[0]:cols[1]:stride]
    v, u = v[:d.shape[0], :d.shape[1]], u[:d.shape[0], :d.shape[1]]
    ok = np.isfinite(d) & (d > depth_range[0]) & (d < depth_range[1])
    if ok.sum() < min_points:
        return None
    Z = d[ok].astype(np.float64)
    P = np.stack([(u[ok] - cx) / fx * Z, (v[ok] - cy) / fy * Z, Z], 1)
    rng = np.random.default_rng(seed)
    # 가설 평가는 표본 3000점으로 충분하다. 노드 주기(15 Hz) 안에서 돌아야 한다.
    S = P[rng.integers(0, len(P), min(len(P), 3000))]
    best_n, best_d, best_cnt = None, None, -1
    for tri in rng.integers(0, len(S), (iters, 3)):
        s = S[tri]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        cnt = int((np.abs(S @ n - n @ s[0]) < inlier_m).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, n @ s[0], cnt
    if best_n is None:
        return None
    best = np.abs(P @ best_n - best_d) < inlier_m
    if best.sum() < 3:
        return None
    Q = P[best]
    c = Q.mean(0)
    # 3×3 공분산의 최소 고유벡터 = 법선. svd(Q) 는 점 수×점 수 행렬을 만들어 한 번에 17초 걸렸다.
    n = np.linalg.eigh(np.cov((Q - c).T))[1][:, 0]
    if n[1] < 0:                          # 광학 y 는 아래 — 법선을 지면 쪽으로
        n = -n
    return {"h": float(n @ c), "pitch_deg": math.degrees(math.asin(max(-1.0, min(1.0, n[2])))),
            "roll_deg": math.degrees(math.atan2(-n[0], n[1])), "inlier": float(best.mean())}


def decode_compressed_depth(data):
    """image_transport compressedDepth (16UC1 png) -> uint16 mm 배열. 앞 12바이트는 설정 헤더."""
    import cv2
    img = cv2.imdecode(np.frombuffer(bytes(data)[12:], np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.dtype != np.uint16:
        return None
    return img
