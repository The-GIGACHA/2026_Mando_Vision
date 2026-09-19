#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traffic_light_go.py — 다수결로 안정된 신호 라벨 -> '갈 수 있는 신호가 보이는가' Bool. ROS 비의존.

★ Bool 에는 '모름' 이 없다. 미검출도 False, 적신호도 False 다.
  False 만 보고는 '빨간불' 인지 '아무것도 못 봤다' 인지 알 수 없으므로,
  이유가 필요하면 /perception/traffic_light (String JSON) 의 label·go_reason 을 같이 볼 것.

★ Bool 하나에 진행방향이 없다. green(직진 신호)과 left(적색+좌회전 화살표)가 둘 다 True 라,
  직진 교차로에서 left 가 True 로 나가면 적신호 진입이 된다. 방향은 제어가 알고 있으므로
  방향 확인이 필요하면 String 의 label 을 같이 볼 것 (2026-09-17 판단팀과 합의한 규약).
"""

GO_LABELS = ("green", "green_left")      # 녹색 램프가 켜져 있으면 방향과 무관하게 진행 신호
ARROW_LABELS = ("left",)                 # 화살표 램프로만 진행 신호가 되는 라벨

# 화살표 램프 폭 / 하우징 폭. 하우징 1.2 에 화살표 약 0.28 (판단팀 제공 치수).
ARROW_RATIO = 0.28 / 1.2
# 화살표가 이 픽셀보다 작으면 red 와 구분이 안 된다 (판단팀 제공 기준).
ARROW_MIN_PX = 12.0


def arrow_px(box, ratio=ARROW_RATIO):
    """bbox 폭에서 추정한 화살표 램프 폭 [px].

    ★ 추측 전제: 모델 bbox 가 '하우징 전체 폭' 을 감싼다고 본다. 라벨링 데이터로
      확인하지 못했다. 램프 하나만 감싸는 라벨이면 이 값이 약 4배 작게 나와
      가까운 left 까지 False 로 떨어진다 (안전한 쪽으로 틀린다).
    """
    if box is None:
        return None
    return max(0.0, float(box[2]) - float(box[0])) * ratio


def decide(label, box, ratio=ARROW_RATIO, min_px=ARROW_MIN_PX):
    """(go, reason). reason 은 String 토픽에 그대로 싣는 짧은 영문 코드."""
    if label is None:
        return False, "no_detection"
    if label in GO_LABELS:
        return True, "green_lamp"
    if label in ARROW_LABELS:
        px = arrow_px(box, ratio)
        if px is None:
            return False, "arrow_no_box"
        # red 를 left 로 읽으면 적신호 진입이고, left 를 red 로 읽으면 정지로 끝난다.
        # 대가가 비대칭이라 화살표를 분해할 수 없는 거리의 left 는 정지 쪽으로 떨어뜨린다.
        if px < min_px:
            return False, "arrow_too_small"
        return True, "arrow"
    return False, "stop_lamp"
