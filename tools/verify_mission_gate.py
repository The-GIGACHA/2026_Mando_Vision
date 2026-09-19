#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_mission_gate.py — 미션 게이트가 실제 용인 맵에서 켜고 끄는 자리를 ROS 없이 확인.

  python3 tools/verify_mission_gate.py

맵은 fma_15_control/global_path/map 의 JSON 을 그대로 읽는다 (GRM 이 /global_active_map 으로
알려 주는 파일과 같다). 맵 구간이 바뀌면 여기 인덱스도 같이 바꿔야 한다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils"))

from mission_gate import MissionGate      # noqa: E402

MAP = os.path.normpath(os.path.join(HERE, "..", "..", "fma_15_control", "global_path", "map"))
FAILS = []


def check(cond, msg):
    print("  %s  %s" % ("PASS" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def at(gate, name, idx, now=100.0):
    gate.set_map(os.path.join(MAP, name))
    gate.set_idx(idx, now)
    return gate.check(now)


print("[1] 신호등 게이트 {TRAFFIC_LIGHT: 15 m}")
tl = MissionGate({"TRAFFIC_LIGHT": 15.0})
a, i = at(tl, "mando_map_1.json", 300)
check(not a and i["mode"] == "off", "map_1[300] (신호 구간 356 에서 멀다) → off")
a, i = at(tl, "mando_map_1.json", 335)
check(a and i["mode"] == "ahead" and i["mission"] == "TRAFFIC_LIGHT",
      "map_1[335] → ahead %s m" % i["dist"])
a, i = at(tl, "mando_map_1.json", 360)
check(a and i["mode"] == "zone", "map_1[360] → zone")
a, i = at(tl, "mando_map_1.json", 380)
check(not a, "map_1[380] (신호 구간 356~375 지남) → off")
a, i = at(tl, "mando_map_3_rere.json", 240)
check(a and i["mode"] == "ahead", "map_3_rere[240] (좌회전 신호 247 앞) → ahead %s m" % i["dist"])

print("[2] 장애물 모델 게이트 {S_COURSE: 0, SUDDEN_STOP: 10, SIGNAL_VEHICLE: 10}")
ob = MissionGate({"S_COURSE": 0.0, "SUDDEN_STOP": 10.0, "SIGNAL_VEHICLE": 10.0})
a, i = at(ob, "mando_map_1.json", 440)
check(not a, "map_1[440] (S코스 450 앞 4 m) → off — S코스는 구간 안에서만")
a, i = at(ob, "mando_map_1.json", 455)
check(a and i["mission"] == "S_COURSE" and ob.in_zone("S_COURSE", 100.0) is True,
      "map_1[455] → S_COURSE zone")
a, i = at(ob, "mando_map_3_rere.json", 425)
check(a and i["mode"] == "ahead" and i["mission"] == "SUDDEN_STOP",
      "map_3_rere[425] → 돌발 ahead %s m" % i["dist"])
a, i = at(ob, "seam_3_4.json", 5)
check(a and i["mode"] == "zone" and i["mission"] == "SUDDEN_STOP", "seam_3_4[5] (돌발 한가운데 이음새) → zone")
a, i = at(ob, "mando_map_5_re.json", 40)
check(a and i["mission"] == "SIGNAL_VEHICLE", "map_5_re[40] → 신호제어차량 ahead %s m" % i["dist"])
a, i = at(ob, "mando_map_2.json", 112)
check(not a and ob.in_zone("S_COURSE", 100.0) is False, "map_2[112] (직각주차) → off, S코스 밖")

print("[3] GRM 이 없으면 항상 켜기")
g = MissionGate({"TRAFFIC_LIGHT": 15.0})
a, i = g.check(100.0)
check(a and i["mode"] == "no_grm" and g.in_zone("TRAFFIC_LIGHT", 100.0) is None, "idx 미수신 → no_grm, 켜짐")
at(g, "mando_map_1.json", 300, now=100.0)
a, i = g.check(103.0)
check(a and i["mode"] == "no_grm", "idx 가 3초 끊김 (stale 2 s) → no_grm, 켜짐")
g = MissionGate({"TRAFFIC_LIGHT": 15.0})
a, i = at(g, "없는_맵.json", 10)
check(a and i["mode"] == "no_map", "맵을 못 읽음 → no_map, 켜짐")

print()
if FAILS:
    print("실패 %d개" % len(FAILS))
    sys.exit(1)
print("전부 통과")
