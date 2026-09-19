#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obstacle_ground.py — 2D 검출 -> base_link 지면 좌표 + 트래킹 + 시간 투표. ROS 비의존.

  배치 위치:  mando_vision_2026/utils/obstacle_ground.py
  사용처:    src/ground_markers.py (노드),  tools/verify_ground_markers.py (ROS 없이 검증)

★ 비전은 좌표만 냅니다. 좌/우 판정·경로 선택은 제어팀 몫이라 여기에는
  side / sigma 가 없습니다. side 는 '차체 중심선' 기준인데, 차가 이미 오른쪽
  경로를 달리고 있으면 그 경로 위 장애물은 차 정면(y≈0)에 와서 side 가
  L/R 사이를 무의미하게 오갑니다.

────────────────────────────────────────────────────────────────────────
좌표
  x, y   bbox 밑변 양 끝 2점만 지면 투영한 평균.  width = |Yl - Yr|
         ★ 윗변(y1)은 지면 좌표로 쓰지 않습니다. 윗변은 공중에 떠 있어서
           광선이 더 먼 지면에 꽂힙니다 (5.0 m 콘의 윗변 -> 8.50 m, 70% 오차).
  z      '물체가 수직으로 서 있다' 가정의 높이.
           Xb = 밑변 중앙의 수평거리,  Xt = 윗변 중앙을 '일부러' 지면 투영한 수평거리
           z  = h * (1 - Xb / Xt)     카메라(높이 h) -> Xt 광선이 Xb 에서 지나는 높이
         Xt 가 없거나 Xt <= Xb 면 None.
  x_err  pitch 를 ±pitch_tol_deg 흔들어 to_ground 를 다시 부른 x 의 폭.
         근사식이 아니라 실제 투영이라, 원거리에서 급격히 커지는 모양이 그대로 나옵니다.
         흔든 쪽이 수평선을 넘어 투영에 실패하면 None (= 추정 불가, 매우 부정확).
         ★ 거리로 거르지 않습니다. 전부 내보내고 x_err 만 붙입니다.

★ 가시 지면 최근접은 base_link 2.70 m 입니다. 그보다 가까운 물체는 밑동이
  화면 아래로 잘려 전부 2.70 m 로 뭉개지고 경고도 안 납니다.
  box 의 y2 가 이미지 높이-1 이면 잘린 것입니다. 그래서 box(픽셀)를 계속 싣습니다.

────────────────────────────────────────────────────────────────────────
트래킹 / 투표 — 소스 구분 없이 지면 (x, y) 하나로 합칩니다
  · 매 주기 '새로 도착한' 프레임들의 검출을 모아 기존 트랙과 최근접 매칭 (gate 이내)
  · 트랙은 자기를 마지막으로 본 소스가 새 프레임을 냈는데 못 찾았을 때만 miss 입니다.
    소스마다 주기가 달라서(obstacle 15 Hz, cone 10 Hz), 이번 주기에 안 온 소스
    때문에 표가 깎이면 안 됩니다.
  · 같은 주기에 다른 소스가 같은 자리를 보면 새 트랙을 만들지 않고 흡수합니다.
  · votes = 최근 vote_window 번의 관측 기회 중 본 횟수,  stable = votes >= vote_min
    ★ stable=false 도 발행합니다. 버리지 않고 제어가 고릅니다.
"""
import math
from collections import deque

from ground import Ground

FRAME_ID = "base_link"


def corners(cx, cy, sx, sy):
    """중심+크기 -> (x1, y1, x2, y2)"""
    return cx - sx / 2.0, cy - sy / 2.0, cx + sx / 2.0, cy + sy / 2.0


def from_detection2d(det, names, labels):
    """vision_msgs/Detection2D -> (label, conf, x1, y1, x2, y2).
    labels 에 없는 라벨(LED 패널 등)이면 None."""
    if not det.results:
        return None
    hyp = det.results[0]
    label = names.get(int(hyp.id))
    if label is None or label not in labels:
        return None
    b = det.bbox
    x1, y1, x2, y2 = corners(b.center.x, b.center.y, b.size_x, b.size_y)
    return label, float(hyp.score), x1, y1, x2, y2


class Projector(object):
    """bbox -> {x, y, z, width, x_err}.  지면 투영은 전부 utils/ground.py 로 합니다."""

    def __init__(self, G, pitch_tol_deg=0.5):
        self.G = G
        pitch = math.degrees(G.th)
        self.G_lo = Ground(G.fx, G.fy, G.cx, G.cy, G.h, pitch - pitch_tol_deg,
                           G.x_off, G.y_off)
        self.G_hi = Ground(G.fx, G.fy, G.cx, G.cy, G.h, pitch + pitch_tol_deg,
                           G.x_off, G.y_off)

    def project(self, x1, y1, x2, y2):
        """밑변이 수평선 위라 지면에 못 놓으면 None."""
        G = self.G
        gl = G.to_ground(x1, y2)             # 밑변 2점만 지면 투영
        gr = G.to_ground(x2, y2)
        if gl is None or gr is None:
            return None
        (Xl, Yl), (Xr, Yr) = gl, gr

        u_c = (x1 + x2) * 0.5
        z = None
        gb = G.to_ground(u_c, y2)
        gt = G.to_ground(u_c, y1)            # 일부러 지면 투영 — 높이 계산에만 씀
        if gb is not None and gt is not None:
            Xb = gb[0] - G.x_off
            Xt = gt[0] - G.x_off
            if Xt > Xb:
                z = G.h * (1.0 - Xb / Xt)

        lo = self.G_lo.to_ground(u_c, y2)
        hi = self.G_hi.to_ground(u_c, y2)
        x_err = abs(hi[0] - lo[0]) if (lo is not None and hi is not None) else None

        return {"x": (Xl + Xr) * 0.5, "y": (Yl + Yr) * 0.5, "z": z,
                "width": abs(Yl - Yr), "x_err": x_err}


def _r(v, nd=3):
    return None if v is None else round(v, nd)


class _Track(object):
    def __init__(self, tid, ob, window):
        self.id = tid
        self.ob = ob                          # 마지막으로 매칭된 관측
        self.misses = 0
        self.hist = deque([1], maxlen=window)

    @property
    def votes(self):
        return sum(self.hist)


class GroundFusion(object):

    def __init__(self, projector, gate_m=0.8, miss_max=5, vote_window=5,
                 vote_min=3, source_timeout=0.5, height_expect=None,
                 height_tol=0.20, image_height=720):
        self.proj = projector
        self.gate = float(gate_m)
        self.miss_max = int(miss_max)
        self.vote_window = int(vote_window)
        self.vote_min = int(vote_min)
        self.timeout = float(source_timeout)
        self.height_expect = dict(height_expect or {})
        self.height_tol = float(height_tol)
        self.image_height = int(image_height)
        self.tracks = []
        self._next_id = 0
        self._latest = {}                     # key -> (도착 시각, image_stamp)

    # ── 주기 ─────────────────────────────────────────────────────
    def step(self, frames, now):
        """
        frames : 이번 주기에 '새로 도착한' 프레임만  [(key, source, image_stamp, dets)]
                   key     소스 구분자(토픽). 같은 section 을 카메라 두 대가 쓸 수 있어서
                   source  페이로드의 "source" 값 (classes.yaml 섹션명)
                   dets    [(label, conf, x1, y1, x2, y2)]  라벨 필터가 끝난 것
        now    : 초. 도착 시각이자 age_ms 기준

        반환 (payload, warnings)
          payload  stamp 는 없습니다 (HeartbeatPublisher 가 붙입니다)
          warnings [(label, z, expected)]  높이 자가검증 실패
        """
        obs, warnings = [], []
        for key, source, stamp, dets in frames:
            self._latest[key] = (now, stamp)
            for label, conf, x1, y1, x2, y2 in dets:
                g = self.proj.project(x1, y1, x2, y2)
                if g is None:
                    continue
                g.update(key=key, label=label, source=source, conf=conf,
                         box=(x1, y1, x2, y2))
                obs.append(g)
                w = self._check_height(g)
                if w is not None:
                    warnings.append(w)

        fresh = self._fresh(now)
        # 끊긴 소스의 트랙은 버립니다. 안 그러면 표가 깎일 기회가 없어 영원히 남습니다.
        self.tracks = [t for t in self.tracks if t.ob["key"] in fresh]
        self._associate(obs, set(f[0] for f in frames))
        return self._payload(now, fresh), warnings

    def _fresh(self, now):
        return {k: st for k, (t, st) in self._latest.items()
                if now - t <= self.timeout}

    # ── 트래킹 + 투표 ────────────────────────────────────────────
    def _associate(self, obs, reported):
        pairs = []
        for ti, t in enumerate(self.tracks):
            for oi, o in enumerate(obs):
                d = math.hypot(t.ob["x"] - o["x"], t.ob["y"] - o["y"])
                if d <= self.gate:
                    pairs.append((d, ti, oi))
        pairs.sort(key=lambda p: p[0])

        took, used = {}, set()
        for _d, ti, oi in pairs:
            if oi in used:
                continue
            if ti not in took:
                took[ti] = obs[oi]
                used.add(oi)
            elif took[ti]["key"] != obs[oi]["key"]:
                used.add(oi)                  # 같은 주기, 다른 소스, 같은 자리 -> 흡수

        survivors = []
        for ti, t in enumerate(self.tracks):
            if ti in took:
                t.ob = took[ti]
                t.misses = 0
                t.hist.append(1)
            elif t.ob["key"] in reported:     # 볼 수 있던 소스가 왔는데 못 봄
                t.misses += 1
                t.hist.append(0)
                if t.misses > self.miss_max:
                    continue
            survivors.append(t)

        for oi, o in enumerate(obs):
            if oi not in used:
                survivors.append(_Track(self._next_id, o, self.vote_window))
                self._next_id += 1
        self.tracks = survivors

    # ── 발행 형태 ────────────────────────────────────────────────
    def _payload(self, now, fresh):
        items = []
        for t in self.tracks:
            # 잠깐 놓친 트랙은 ID·표만 유지하고 목록에서는 뺍니다 (낡은 좌표 금지)
            if t.misses or t.ob["key"] not in fresh:
                continue
            o = t.ob
            x1, y1, x2, y2 = o["box"]
            votes = t.votes
            items.append({
                "id": t.id, "label": o["label"], "source": o["source"],
                "conf": _r(o["conf"]),
                "x": _r(o["x"]), "y": _r(o["y"]), "z": _r(o["z"]),
                "width": _r(o["width"]), "x_err": _r(o["x_err"]),
                "votes": votes, "stable": votes >= self.vote_min,
                "box": [int(x1), int(y1), int(x2), int(y2)],
            })
        items.sort(key=lambda it: it["x"])

        stamps = [s for s in fresh.values() if s is not None]
        istamp = max(stamps) if stamps else None

        # ★ 고정 키 집합을 먼저 만들고 값만 채웁니다. 검출이 없어도 키가 같아야
        #   제어 쪽 파서가 분기 없이 읽습니다.
        d = {"detected": False, "n": 0, "frame_id": FRAME_ID, "items": [],
             "image_stamp": None, "age_ms": None}
        d["detected"] = bool(items)
        d["n"] = len(items)
        d["items"] = items
        if istamp is not None:
            d["image_stamp"] = istamp
            d["age_ms"] = round((now - istamp) * 1000.0, 1)
        return d

    # ── 자가검증 ─────────────────────────────────────────────────
    def _check_height(self, g):
        """물체 실제 높이를 아니까, z 가 틀리면 거리가 틀렸다는 신호입니다."""
        exp = self.height_expect.get(g["label"])
        if exp is None or g["z"] is None:
            return None
        _x1, y1, _x2, y2 = g["box"]
        # 윗변/밑변이 화면 끝에 닿은 박스는 잘려서 z 가 원래 틀립니다.
        # 캘리브레이션 탓이 아니므로 경고하지 않습니다.
        if y1 <= 1.0 or y2 >= self.image_height - 2:
            return None
        if abs(g["z"] - exp) > self.height_tol * exp:
            return g["label"], g["z"], exp
        return None
