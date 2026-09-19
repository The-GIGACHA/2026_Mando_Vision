#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s_course.py — S자 코스 첫 번째 T870 이 차량 기준 우측인가. ROS 비의존.

  사용처: src/obstacle_detector.py,  tools/verify_s_course.py (ROS 없이 검증)

★ S자 코스 장애물은 좌->우 또는 우->좌로만 놓인다. 첫 번째의 좌/우만 알면
  나머지가 결정되므로, 첫 판정을 커밋한 뒤에는 절대 뒤집지 않는다.
  횡방향 기동 중에 경로가 바뀌는 것이 접촉으로 가는 전형적 경로다.

★ 기준은 '판정 시점의 차량 중심선(base_link y=0)' 이다. 차가 이미 한쪽 레인으로
  옮겨 탄 뒤에 판정하면 그 레인 위 장애물이 y≈0 에 와서 좌/우가 뒤집힐 수 있다.
  그래서 중심선 근처(|y| < min_abs_y)는 표로 세지 않는다.

Bool 규약 (2026-09-17 합의): 기본 False, 커밋 후 우측 True / 좌측 False.
  ★ False 에는 '아직 모름' 과 '좌측 확정' 이 섞여 있다. 구분이 필요하면
    상세 String(/perception/s_course) 의 state 를 볼 것.
"""
from collections import deque

LABEL = "T870"


class SCourseJudge(object):

    def __init__(self, ground, max_x=10.0, min_abs_y=0.10, vote_window=7,
                 vote_min=5, image_height=720):
        self.G = ground
        # 먼 거리는 yaw 오차가 횡방향으로 그대로 들어온다 (오차 ≈ 거리 × tan(yaw)).
        # 결정 거리를 짧게 잡을수록 그 위험이 줄어든다.
        self.max_x = float(max_x)
        self.min_abs_y = float(min_abs_y)
        self.vote_min = int(vote_min)
        self.image_height = int(image_height)
        self.votes = deque(maxlen=int(vote_window))
        self.committed = None            # None | "RIGHT" | "LEFT"
        self.last = None                 # 마지막 프레임의 최근접 T870 관측

    def reset(self):
        self.votes.clear()
        self.committed = None
        self.last = None

    def nearest(self, dets):
        """이번 프레임에서 지면 거리가 가장 가까운 T870 관측. 없으면 None.

        검출 리스트 순서는 신뢰도·모델 내부 순서라 거리와 무관하다. 그래서 전부 투영해 고른다.
        """
        best = None
        for d in dets:
            if d.get("name") != LABEL:
                continue
            x1, _y1, x2, y2 = d["box"]
            # 밑동이 화면 아래로 잘린 박스는 접지점이 아니라 화면 끝을 투영하게 된다.
            if y2 >= self.image_height - 2:
                continue
            # 입체는 IPM 에서 늘어나므로 지면과 닿은 하단 모서리 중심 한 점만 투영한다.
            g = self.G.to_ground((x1 + x2) * 0.5, float(y2))
            if g is None:
                continue
            X, Y = g
            if best is None or X < best["x"]:
                best = {"x": X, "y": Y, "conf": d.get("conf"), "box": d["box"]}
        return best

    def update(self, dets):
        """프레임 한 장 반영. 반환: 상세 상태 dict (String 토픽용)."""
        ob = self.nearest(dets) if dets is not None else None
        self.last = ob
        vote = None
        if self.committed is None and ob is not None and ob["x"] <= self.max_x:
            if abs(ob["y"]) >= self.min_abs_y:
                vote = "LEFT" if ob["y"] > 0 else "RIGHT"
            self.votes.append(vote)
            n_r = sum(1 for v in self.votes if v == "RIGHT")
            n_l = sum(1 for v in self.votes if v == "LEFT")
            if n_r >= self.vote_min and n_r > n_l:
                self.committed = "RIGHT"
            elif n_l >= self.vote_min and n_l > n_r:
                self.committed = "LEFT"
        return self.status(vote)

    @property
    def right(self):
        """Bool 토픽 값. 커밋 전과 좌측 확정이 둘 다 False 다."""
        return self.committed == "RIGHT"

    def status(self, vote=None):
        if self.committed is not None:
            state = "COMMITTED"
        elif self.votes:
            state = "VOTING"
        else:
            state = "WAITING"
        ob = self.last
        return {
            "state": state,
            "side": self.committed,
            "right": self.right,
            "vote": vote,
            "votes_right": sum(1 for v in self.votes if v == "RIGHT"),
            "votes_left": sum(1 for v in self.votes if v == "LEFT"),
            "window": self.votes.maxlen, "vote_min": self.vote_min,
            "nearest": None if ob is None else {
                "x": round(ob["x"], 2), "y": round(ob["y"], 3),
                "conf": None if ob["conf"] is None else round(ob["conf"], 3),
                "box": [int(v) for v in ob["box"]]},
            "max_x": self.max_x,
        }
