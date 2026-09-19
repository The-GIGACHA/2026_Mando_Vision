#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ground.py — 픽셀 <-> 지면 좌표 (평평한 지면 가정). ROS 비의존.

★ 지면 위의 점만 유효합니다.
  호모그래피/IPM 은 지면 평면 위의 점만 올바르게 옮깁니다. 지면에서 솟은
  입체(T870 차체 등)는 카메라 반대 방향으로 늘어납니다. 물체의 bbox 중심이
  아니라 '지면과 닿는 선', 즉 bbox 하단 모서리 중심을 넣어야 합니다.

★ 횡방향은 정밀하고 종방향은 무너집니다 (2026-09-12 실측 K 기준).
      거리    가로 1px      세로 1px
       5 m     0.8 cm        3.8 cm
      10 m     1.6 cm       15.2 cm
      20 m     3.1 cm       60.2 cm
  좌/우 판정(횡방향)은 30 m 에서도 5 cm 해상도지만, 거리(종방향)는 20 m 만
  가도 1 px 이 60 cm 입니다. 거리는 뎁스로 교차검증하십시오.
  같은 이유로 횡방향은 차량 pitch 흔들림에 둔감합니다.

★ yaw 오차는 횡방향에 직접 들어갑니다: 오차 ≈ 거리 × tan(yaw).
  yaw 3도면 20 m 에서 1.05 m 로, 판정을 뒤집을 수 있는 크기입니다.
  결정 거리를 짧게 가져갈수록 이 위험이 줄어듭니다.
"""
import math


class Ground(object):

    def __init__(self, fx, fy, cx, cy, h, pitch_deg, x_off=0.0, y_off=0.0):
        """h: 광학중심의 지면 높이 [m]
        pitch_deg: 아래를 보면 +, 위를 보면 - (D455 는 -2.75)
        x_off, y_off: base_link 원점에서 광학중심까지 [m], y 는 좌측이 +
        """
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.h = h
        self.th = math.radians(pitch_deg)
        self.x_off, self.y_off = x_off, y_off
        # 지면이 무한대로 가는 행. 이 위쪽은 하늘이라 역변환이 발산합니다.
        self.v_horizon = cy + fy * math.tan(self.th)

    # ── 역변환: 픽셀 -> 지면 ───────────────────────────────────────────
    def to_ground(self, u, v, margin_px=5.0):
        """접지점 픽셀 -> base_link 기준 (X 전방, Y 좌측+). 실패 시 None."""
        if v <= self.v_horizon + margin_px:
            return None
        p = (v - self.cy) / self.fy
        den = p * math.cos(self.th) + math.sin(self.th)
        if den <= 1e-9:
            return None
        X = self.h * (math.cos(self.th) - p * math.sin(self.th)) / den
        if X <= 0.0 or X > 200.0:
            return None
        q = (u - self.cx) / self.fx
        Y = -q * (X * math.cos(self.th) + self.h * math.sin(self.th))
        return X + self.x_off, Y + self.y_off

    # ── 정변환: 지면 -> 픽셀 (검증·시각화용) ───────────────────────────
    def to_pixel(self, X, Y):
        Xc, Yc = X - self.x_off, Y - self.y_off
        den = Xc * math.cos(self.th) + self.h * math.sin(self.th)
        if den <= 1e-9:
            return None
        u = self.cx - self.fx * Yc / den
        v = self.cy + self.fy * (self.h * math.cos(self.th)
                                 - Xc * math.sin(self.th)) / den
        return u, v

    def lateral_res(self, X):
        """해당 거리에서 가로 1 px 이 몇 m 인지."""
        return max(X - self.x_off, 0.1) / self.fx

    # ── 판정 ───────────────────────────────────────────────────────────
    def judge_box(self, box, jitter_frac=0.10, min_sigma_m=0.02):
        """bbox -> 좌/우 관측. 판정이 아니라 '관측 + 근거'를 돌려줍니다.

        sigma 는 검출기 지터를 bbox 폭의 일정 비율로 본 추정 오차입니다.
        부호만 보면 '중앙 근처'와 '명백히 한쪽'을 구분할 수 없으므로,
        다운스트림은 side 가 아니라 sigma 를 보고 보수적으로 굴 수 있습니다.
        """
        x1, y1, x2, y2 = box
        g = self.to_ground((x1 + x2) * 0.5, float(y2))   # 하단 중심 = 접지점
        if g is None:
            return None
        X, Y = g
        sigma = max(abs(x2 - x1) * jitter_frac * self.lateral_res(X),
                    min_sigma_m)
        return {"x": round(X, 2), "y": round(Y, 3),
                "side": "L" if Y > 0 else "R",
                "sigma": round(abs(Y) / sigma, 2)}


def from_params(get, prefix="~ground/"):
    """rospy.get_param 같은 콜러블에서 Ground 를 만듭니다.

    기본값은 2026-09-12 실측입니다 (camera_info 1280x720 + URDF).
    ★ URDF 를 고치면 여기 기본값도 같이 고치십시오. 두 군데가 어긋나도
      에러가 나지 않고 좌표만 조용히 틀어집니다.
    """
    p = prefix
    return Ground(
        fx=get(p + "fx", 644.1228), fy=get(p + "fy", 643.3088),
        cx=get(p + "cx", 646.2876), cy=get(p + "cy", 354.2151),
        h=get(p + "cam_height", 1.000),        # front_z 0.850 + wheel_radius 0.150
        # 2026-09-17 -1.0 -> -2.75 (09-16 송도 bag 노면 뎁스 평면). URDF front_pitch 와 같이 바꿨다.
        pitch_deg=get(p + "cam_pitch_deg", -2.75),
        x_off=get(p + "cam_x", 0.725),
        y_off=get(p + "cam_y", -0.060))        # RGB 광축은 차체 중심 우측 6cm
