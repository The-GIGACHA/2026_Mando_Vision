#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sign_car.py — 신호제어차량 LED 패널에서 차로 지시를 뽑아냅니다.

  배치 위치:  mando_vision_2026/utils/sign_car.py

────────────────────────────────────────────────────────────────────────
문제

obstacle.pt 는 패널을 네 클래스로 냅니다.

    left_go(↓)   left_X    right_go(↓)   right_X

두 패널은 상보적이라 실제 상태는 둘뿐입니다.

    left_go + right_X  ->  좌측 차로로
    left_X  + right_go ->  우측 차로로

그런데 right_X 가 화면 끝의 엉뚱한 것에서 간헐적으로 뜹니다.
X 는 붉은 사선 두 개라 배경의 붉은 구조물·표지판과 섞이기 쉽고,
↓ 는 초록 화살표라 형태와 색이 훨씬 뚜렷합니다.

────────────────────────────────────────────────────────────────────────
그래서 세 겹으로 거릅니다

 ① 차체 게이트 (가장 강력)
    패널은 신호제어차량 몸통에 붙어 있습니다. traffic_car 박스 안에
    들어있지 않은 패널 검출은 버립니다. '화면 끝에서 뜨는' 오검출은
    여기서 대부분 죽습니다.

 ② go 우선 + 비대칭 임계
    ↓ 는 낮은 임계로 받고, X 는 높은 임계를 요구합니다.
    ↓ 가 하나라도 살아 있으면 그것만으로 지시를 확정하고
    X 는 쳐다보지 않습니다. X 는 ↓ 가 전혀 없을 때만 씁니다.

 ③ 시간 다수결
    최근 N 프레임 중 K 프레임 이상 같은 지시여야 확정합니다.

────────────────────────────────────────────────────────────────────────
★ 판단팀에 넘기는 것은 '명령'이 아니라 '관측 + 근거'입니다

  command      : LEFT / RIGHT / None
  basis        : "go" 면 ↓ 를 직접 봤다,  "x" 면 X 만 보고 상보 추론했다
  confidence   : high / medium / low
  both_seen    : 상보 쌍(↓ 와 X)을 둘 다 봤다
  conflict     : 모순된 검출이 있었다

basis 가 "x" 면 판단팀이 더 보수적으로 처리할 수 있습니다.
인지가 단정해서 넘기면 판단팀은 그 불확실성을 영영 알 수 없습니다.
"""

from collections import Counter, deque

PANELS = ("left_go", "left_X", "right_go", "right_X")
HOST = "traffic_car"

# 상보 규칙: 하나만 봐도 지시가 결정됩니다
COMPLEMENT = {
    "left_go": "LEFT",     # 좌측 개방  -> 좌측 차로로
    "right_X": "LEFT",     # 우측 폐쇄  -> 좌측 차로로
    "right_go": "RIGHT",
    "left_X": "RIGHT",
}
PAIR = {"LEFT": ("left_go", "right_X"), "RIGHT": ("right_go", "left_X")}


class SignCarParams(object):
    def __init__(self):
        # ── ② go 우선: ↓ 는 관대하게, X 는 엄격하게 ──────────────
        self.conf_go = 0.25
        self.conf_x = 0.45

        # ── ① 차체 게이트 ───────────────────────────────────────
        self.require_host = True   # 차체를 못 보면 패널을 안 믿습니다
        self.host_margin = 0.12   # 차체 박스를 이 비율만큼 넓혀서 판정
        self.min_inside = 0.60   # 패널 면적의 이 비율 이상이 안에 들어와야

        # ── ③ 시간 다수결 ───────────────────────────────────────
        self.vote_n = 7
        self.vote_k = 4


def _inside_ratio(panel, host, margin):
    """패널 박스 면적 중 (확장된) 차체 박스와 겹치는 비율."""
    px1, py1, px2, py2 = panel
    hx1, hy1, hx2, hy2 = host
    mw = (hx2 - hx1) * margin
    mh = (hy2 - hy1) * margin
    hx1, hy1, hx2, hy2 = hx1 - mw, hy1 - mh, hx2 + mw, hy2 + mh

    ix1, iy1 = max(px1, hx1), max(py1, hy1)
    ix2, iy2 = min(px2, hx2), min(py2, hy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    pa = max((px2 - px1) * (py2 - py1), 1e-9)
    return (iw * ih) / pa


def decide(dets, p):
    """
    dets : [{"name": str, "conf": float, "box": (x1,y1,x2,y2)}, ...]
    p    : SignCarParams

    반환 dict — 아래 obstacle_detector.py 가 그대로 JSON 으로 발행합니다.
    """
    hosts = [d["box"] for d in dets if d["name"] == HOST]
    host_conf = max([d["conf"] for d in dets if d["name"] == HOST] or [0.0])

    best = {k: 0.0 for k in PANELS}
    dropped = Counter()

    for d in dets:
        n = d["name"]
        if n not in PANELS:
            continue
        thr = p.conf_go if n.endswith("_go") else p.conf_x
        if d["conf"] < thr:
            dropped["conf<%.2f" % thr] += 1
            continue
        if p.require_host:
            if not hosts:
                dropped["차체 미검출"] += 1
                continue
            if max(_inside_ratio(d["box"], h, p.host_margin)
                   for h in hosts) < p.min_inside:
                dropped["차체 밖"] += 1
                continue
        if d["conf"] > best[n]:
            best[n] = d["conf"]

    go_l, go_r = best["left_go"], best["right_go"]
    x_l, x_r = best["left_X"], best["right_X"]

    conflict = False
    # 같은 쪽에 ↓ 와 X 가 동시에 = 모순
    if (go_l > 0 and x_l > 0) or (go_r > 0 and x_r > 0):
        conflict = True

    command = None
    basis = None
    if go_l > 0 or go_r > 0:
        basis = "go"
        if go_l > 0 and go_r > 0:
            conflict = True                       # 양쪽 다 개방은 상보 아님
        command = "LEFT" if go_l >= go_r else "RIGHT"
    elif x_l > 0 or x_r > 0:
        basis = "x"
        if x_l > 0 and x_r > 0:
            conflict = True
        # 상보 추론: 한쪽이 막혔으면 반대쪽으로
        command = COMPLEMENT["left_X"] if x_l >= x_r else COMPLEMENT["right_X"]

    both_seen = False
    if command:
        a, b = PAIR[command]
        both_seen = best[a] > 0 and best[b] > 0

    if command is None:
        confidence = None
    elif conflict:
        confidence = "low"
    elif basis == "go" and both_seen:
        confidence = "high"
    elif basis == "go":
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "command": command,
        "basis": basis,
        "confidence": confidence,
        "both_seen": both_seen,
        "conflict": conflict,
        "host": bool(hosts),
        "host_conf": round(host_conf, 3),
        "panels": {k: (round(v, 3) if v > 0 else None) for k, v in best.items()},
        "dropped": dict(dropped),
    }


class SignCarVoter(object):
    """최근 N 프레임 다수결. 단발 오검출이 지시로 나가는 것을 막습니다."""

    def __init__(self, p):
        self.p = p
        self.buf = deque(maxlen=p.vote_n)

    def update(self, obs):
        self.buf.append(obs)
        c = Counter(o["command"] for o in self.buf if o["command"])
        if not c:
            return None, 0
        cmd, n = c.most_common(1)[0]
        if n < self.p.vote_k:
            return None, n
        return cmd, n

    def best_evidence(self, cmd):
        """확정된 지시를 뒷받침하는 관측 중 가장 강한 것을 고릅니다."""
        rank = {"high": 3, "medium": 2, "low": 1, None: 0}
        cands = [o for o in self.buf if o["command"] == cmd]
        if not cands:
            return None
        return max(cands, key=lambda o: rank[o["confidence"]])
