#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mission_gate.py — 전역경로 미션 구역에서만 인지를 켠다.

  사용처: src/traffic_light_detector.py, src/obstacle_detector.py, tools/verify_mission_gate.py

★ 인지는 원래 이미지만 구독한다. 이 게이트만 예외로 제어의 GRM 토픽을 읽는다
  (2026-09-17 판단·제어팀 합의). 미션이 아닌 곳에서 비슷한 물체를 보고 토픽을 쏘는 것,
  특히 한 번 커밋하면 고정되는 S자 코스 판정이 구간 밖 오검출로 오염되는 것을 막기 위해서다.

입력 (제어 GRM)
  /global_active_map   std_msgs/String   지금 달리는 구간 맵 JSON 경로 (latch)
  /global_nearest_idx  std_msgs/Int32    그 맵 안의 현재 인덱스

판정
  missions = {미션 이름: 앞쪽 거리[m]}. 현재 인덱스가 그 미션 구간 안이거나,
  앞쪽 거리 안에 그 미션 구간 시작점이 있으면 활성.
  ★ 앞쪽 탐색은 지금 맵 파일 안에서만 한다. 다음 파일 첫머리의 구간은 그 파일로
    넘어간 뒤부터 보인다.

★ GRM 토픽이 stale_sec 넘게 안 오면 '항상 켜기' 로 떨어진다 (단독 시험·배그 재생·
  제어 미기동). 본경기에 GRM 이 죽어도 신호를 놓치지 않게 하려는 선택이고,
  그 대가로 그동안은 구간 밖 오검출을 막지 못한다. 그래서 경고를 계속 낸다.
"""
import json
import math


def load_map(path):
    """맵 JSON -> (missions, 누적거리[m]). 점 순서는 키 숫자 순서."""
    with open(path) as f:
        d = json.load(f)
    keys = sorted((k for k in d if str(k).isdigit()), key=int)
    missions = [d[k].get("mission", "") for k in keys]
    s = [0.0]
    for a, b in zip(keys, keys[1:]):
        s.append(s[-1] + math.hypot(d[b]["x"] - d[a]["x"], d[b]["y"] - d[a]["y"]))
    return missions, s


class MissionGate(object):

    def __init__(self, missions, stale_sec=2.0):
        self.missions = {str(k): float(v) for k, v in dict(missions).items()}
        self.stale_sec = float(stale_sec)
        self.map_path = None
        self.map_missions = []
        self.map_s = []
        self.idx = None
        self.idx_t = None
        self.load_error = None

    # ── 입력 ─────────────────────────────────────────────────────
    def set_map(self, path):
        if not path or path == self.map_path:
            return
        try:
            self.map_missions, self.map_s = load_map(path)
            self.load_error = None
        except (IOError, OSError, ValueError, KeyError) as e:
            # 맵을 못 읽으면 구간을 모르는 것이므로 GRM 미수신과 같이 '항상 켜기' 로 간다
            self.map_missions, self.map_s = [], []
            self.load_error = "%s: %s" % (path, e)
        self.map_path = path

    def set_idx(self, idx, now):
        self.idx = int(idx)
        self.idx_t = float(now)

    # ── 판정 ─────────────────────────────────────────────────────
    def check(self, now):
        """(active, info). info 는 상세 String 에 그대로 싣는다.

        info["mode"]
          zone      지금 미션 구간 안
          ahead     앞쪽 거리 안에 미션 구간 시작점이 있다
          off       미션 구역 밖 -> 추론·발행을 끈다
          no_grm    GRM 토픽이 없거나 오래됐다 -> 항상 켜기
          no_map    맵을 못 읽었다 -> 항상 켜기
        """
        if self.idx is None or self.idx_t is None or now - self.idx_t > self.stale_sec:
            return True, {"mode": "no_grm", "mission": None, "dist": None}
        if not self.map_missions:
            return True, {"mode": "no_map", "mission": None, "dist": None,
                          "error": self.load_error}
        n = len(self.map_missions)
        i = max(0, min(self.idx, n - 1))
        here = self.map_missions[i]
        if here in self.missions:
            return True, {"mode": "zone", "mission": here, "dist": 0.0}
        reach = max(self.missions.values()) if self.missions else 0.0
        for k in range(i + 1, n):
            d = self.map_s[k] - self.map_s[i]
            if d > reach:
                break
            m = self.map_missions[k]
            if m in self.missions and self.map_missions[k - 1] != m and d <= self.missions[m]:
                return True, {"mode": "ahead", "mission": m, "dist": round(d, 1)}
        return False, {"mode": "off", "mission": None, "dist": None}

    def in_zone(self, mission, now):
        """지금 그 미션 구간 '안' 인가. GRM 이 없으면 None (모름)."""
        if self.idx is None or self.idx_t is None or now - self.idx_t > self.stale_sec:
            return None
        if not self.map_missions:
            return None
        i = max(0, min(self.idx, len(self.map_missions) - 1))
        return self.map_missions[i] == mission


def attach(gate, rospy_mod):
    """rospy 구독을 붙인다. ROS 없는 검증에서는 부르지 않는다."""
    from std_msgs.msg import Int32, String
    rospy_mod.Subscriber("/global_active_map", String,
                         lambda m: gate.set_map(m.data), queue_size=1)
    rospy_mod.Subscriber("/global_nearest_idx", Int32,
                         lambda m: gate.set_idx(m.data, rospy_mod.Time.now().to_sec()),
                         queue_size=1)
