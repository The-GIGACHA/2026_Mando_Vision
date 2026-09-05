#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class_map.py — config/classes.yaml 을 읽고, 실제 모델의 names 와 대조합니다.

2025 코드의 최대 위험은 "코드에 하드코딩된 클래스 순서"와 "실제 가중치의
클래스 순서"가 어긋난 채로 조용히 돌아간 것이었습니다.
이 모듈은 노드 시작 시 둘을 대조하고, 다르면 즉시 죽입니다.
조용히 틀린 판정을 내보내는 것보다 안 뜨는 게 낫습니다.
"""

import os
import re
import yaml
import rospy
import rospkg

_PREFIX = re.compile(r"^\d+_")      # "0_green_left" → "green_left"


def _norm(name):
    """모델 names 의 원본 데이터셋 인덱스 접두어를 제거합니다."""
    return _PREFIX.sub("", str(name)).strip()


def package_path(pkg="mando_vision_2026"):
    return rospkg.RosPack().get_path(pkg)


def load(section, pkg="mando_vision_2026"):
    """classes.yaml 의 한 섹션을 dict 로 반환합니다."""
    path = os.path.join(package_path(pkg), "config", "classes.yaml")
    if not os.path.exists(path):
        rospy.logfatal("classes.yaml 없음: %s", path)
        raise SystemExit(1)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if section not in cfg:
        rospy.logfatal("classes.yaml 에 '%s' 섹션이 없습니다", section)
        raise SystemExit(1)
    sec = cfg[section]
    sec["names"] = {int(k): _norm(v) for k, v in sec.get("names", {}).items()}
    return sec


def verify(model_names, expected, section):
    """
    모델의 실제 names 와 classes.yaml 을 대조합니다.
    불일치면 ROS_FATAL 후 종료 — 조용히 넘어가지 않습니다.
    """
    actual = {int(k): _norm(v) for k, v in dict(model_names).items()}
    if actual == expected:
        rospy.loginfo("[%s] 클래스 매핑 확인: %d개 일치", section, len(actual))
        return actual

    rospy.logfatal("=" * 62)
    rospy.logfatal("[%s] 클래스 매핑 불일치 — 노드를 종료합니다", section)
    rospy.logfatal("=" * 62)
    for i in sorted(set(actual) | set(expected)):
        a, e = actual.get(i, "—"), expected.get(i, "—")
        rospy.logfatal("  %2d  모델:%-18s classes.yaml:%-18s %s",
                       i, a, e, "" if a == e else "  <-- 다름")
    rospy.logfatal("")
    rospy.logfatal("가중치를 바꿨다면 config/classes.yaml 을 먼저 갱신하세요.")
    rospy.logfatal("추출:  python3 tools/check_weights.py <weights 폴더>")
    raise SystemExit(1)


def load_and_verify(section, model_names, pkg="mando_vision_2026"):
    sec = load(section, pkg)
    verify(model_names, sec["names"], section)
    return sec
