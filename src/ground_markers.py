#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ground_markers.py — 2D 검출 여러 소스 -> base_link 지면 좌표 -> /perception/obstacle

  배치 위치:  mando_vision_2026/src/ground_markers.py

  obstacle_detector (obstacle.engine) -> /detect/obstacle ─┐
  object_detector   (cone.engine)     -> /detect/cone     ─┴-> ground_markers
      -> /perception/obstacle               std_msgs/String (JSON)  매 주기   ← 제어팀
      -> /perception/obstacle/viz/markers   MarkerArray             구독자 있을 때만

★ /perception/obstacle 의 유일한 발행자입니다. obstacle_detector 는 더 이상 이
  토픽에 쓰지 않습니다. 둘이 쓰면 제어가 좌표 있는 것/없는 것을 번갈아 받는데
  에러가 안 나서 알아채기 어렵습니다.
★ 비전은 좌표만 냅니다. 좌/우 판정·경로 선택은 제어팀이 합니다.
  계산 규칙(투영·높이·x_err·트래킹·투표)은 utils/obstacle_ground.py 에 있습니다.

파라미터
  ~sources         [{topic, section, labels}, ...]
                     section  classes.yaml 섹션 — id->이름 매핑이 소스마다 다릅니다
                     labels   이 라벨만 씁니다 (LED 패널·traffic_car 등은 여기서 결정)
  ~output_topic    /perception/obstacle
  ~marker_topic    /perception/obstacle/viz/markers
  ~rate_hz 15  ~viz_hz 15  ~source_timeout 0.5 [s]
  ~track_gate_m 0.8  ~track_miss_max 5  ~vote_window 5  ~vote_min 3
  ~ground/*        utils/ground.py from_params (K, 카메라 높이·pitch·장착 위치)
  ~ground/pitch_tol_deg 0.5
  ~height_expect   {라벨: 실제 높이 m}   ~height_tol 0.20   ~image_height 720

/perception/obstacle 페이로드 — 키 집합은 검출이 없어도 항상 같습니다
  {"detected": true, "n": 1, "frame_id": "base_link",
   "items": [{"id": 3, "label": "cone_blue", "source": "cone", "conf": 0.87,
              "x": 4.21, "y": 0.35, "z": 0.45, "width": 0.20,
              "x_err": 0.22, "votes": 5, "stable": true,
              "box": [610, 402, 690, 520]}],
   "image_stamp": 1726400000.045, "age_ms": 78.3, "stamp": 1726400000.123}

  x, y, z, width  base_link [m] (x 전방, y 좌측+). z 는 수직 물체 가정 높이, 못 구하면 null
  x_err           pitch ±pitch_tol_deg 흔들었을 때 x 의 폭 [m]. 원거리일수록 커집니다
  box             원본 이미지 픽셀. y2 가 이미지 높이-1 이면 밑동이 잘려 x 가 2.70 m 로 뭉개진 것
  image_stamp     살아있는 소스가 마지막으로 처리한 영상 중 가장 최근 캡처 시각.
                  검출이 0개여도 영상을 처리했으면 채웁니다 (기존 obstacle_detector 와 같은 의미).
                  null 이면 source_timeout 동안 어느 소스에서도 영상이 안 온 것 = 입력 끊김
  items           이번에 실제로 보인 트랙만. 잠깐 놓친 트랙은 id·votes 를 내부에 유지하고
                  낡은 좌표를 싣지 않으려고 목록에서만 뺍니다
"""

import math
import os
import sys
import threading

import rospy
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray


def _add_utils_to_path():
    """catkin_install_python 이 스크립트를 devel/lib/<pkg>/ 로 복사하므로
    __file__ 기준 '../utils' 가 거기서는 존재하지 않습니다."""
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "..", "utils")
    if not os.path.isdir(cand):
        import rospkg
        cand = os.path.join(
            rospkg.RosPack().get_path("mando_vision_2026"), "utils")
    if cand not in sys.path:
        sys.path.insert(0, cand)


_add_utils_to_path()

import class_map                                                  # noqa: E402
from ground import from_params                                    # noqa: E402
from infer_loop import HeartbeatPublisher, Throttle               # noqa: E402
from obstacle_ground import (FRAME_ID, GroundFusion, Projector,   # noqa: E402
                             from_detection2d)

DEFAULT_SOURCES = [
    {"topic": "/detect/obstacle", "section": "obstacle", "labels": ["T870", "kid"]},
    {"topic": "/detect/cone", "section": "cone",
     "labels": ["cone_blue", "cone_yellow"]},
]
DEFAULT_HEIGHT = {"cone_blue": 0.45, "cone_yellow": 0.45, "T870": 0.40, "kid": 0.60}

COLOR = {"cone_blue": (0.15, 0.35, 1.0), "cone_yellow": (1.0, 0.85, 0.1),
         "T870": (0.25, 0.85, 0.25), "kid": (1.0, 0.5, 0.1)}


class GroundMarkers(object):
    def __init__(self):
        rospy.init_node("ground_markers")
        g = rospy.get_param

        self.G = from_params(rospy.get_param)
        self.rate_hz = float(g("~rate_hz", 15.0))
        self.viz_hz = float(g("~viz_hz", 10.0))

        self._lock = threading.Lock()
        self.sources = [self._make_source(i, s)
                        for i, s in enumerate(g("~sources", DEFAULT_SOURCES))]
        if not self.sources:
            rospy.logfatal("~sources 가 비었습니다")
            raise SystemExit(1)

        pitch_tol = float(g("~ground/pitch_tol_deg", 0.5))
        self.fusion = GroundFusion(
            Projector(self.G, pitch_tol),
            gate_m=float(g("~track_gate_m", 0.8)),
            miss_max=int(g("~track_miss_max", 5)),
            vote_window=int(g("~vote_window", 5)),
            vote_min=int(g("~vote_min", 3)),
            source_timeout=float(g("~source_timeout", 0.5)),
            height_expect=g("~height_expect", DEFAULT_HEIGHT),
            height_tol=float(g("~height_tol", 0.20)),
            image_height=int(g("~image_height", 720)))

        self.out = HeartbeatPublisher(g("~output_topic", "/perception/obstacle"))
        self.marker_pub = rospy.Publisher(
            g("~marker_topic", "/perception/obstacle/viz/markers"),
            MarkerArray, queue_size=1)
        self.marker_throttle = Throttle(self.viz_hz)

        for i, src in enumerate(self.sources):
            rospy.Subscriber(src["topic"], Detection2DArray, self._cb,
                             callback_args=i, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.step)
        rospy.Timer(rospy.Duration(5.0), self.watchdog)
        rospy.loginfo("ground_markers 준비 완료 (%.0f Hz, pitch=%.2f°±%.2f, h=%.3f m) 소스: %s",
                      self.rate_hz, math.degrees(self.G.th), pitch_tol,
                      self.G.h,
                      ", ".join("%s[%s]" % (s["topic"], "/".join(s["labels"]))
                                for s in self.sources))

    def _make_source(self, i, s):
        try:
            topic, section = str(s["topic"]), str(s["section"])
            labels = [str(v) for v in s["labels"]]
        except (KeyError, TypeError):
            rospy.logfatal("~sources[%d] 에 topic / section / labels 가 모두 필요합니다: %r",
                           i, s)
            raise SystemExit(1)
        names = class_map.load(section)["names"]
        unknown = sorted(set(labels) - set(names.values()))
        if not labels or unknown:
            # 오타 난 라벨은 에러 없이 전부 걸러져 '장애물 없음'으로 나갑니다.
            rospy.logfatal("~sources[%d] %s: labels %s 가 classes.yaml '%s' 에 없습니다 "
                           "(있는 이름: %s)", i, topic, unknown or "(비어 있음)",
                           section, sorted(names.values()))
            raise SystemExit(1)
        return {"topic": topic, "section": section, "labels": set(labels),
                "names": names, "msg": None, "n": 0}

    def _cb(self, msg, i):
        with self._lock:
            self.sources[i]["msg"] = msg        # 덮어쓰기 — 최신 프레임만
            self.sources[i]["n"] += 1

    # ── 주기 ─────────────────────────────────────────────────────
    def step(self, _evt):
        now = rospy.Time.now().to_sec()
        frames = []
        for src in self.sources:
            with self._lock:
                msg, src["msg"] = src["msg"], None
            if msg is None:
                continue
            if msg.header.stamp.is_zero():
                # 검출 노드가 이번 주기에 영상을 못 받았습니다 (obstacle_detector 는
                # stamp 0 인 빈 배열로 알립니다). '봤는데 없음' 으로 치지 않습니다.
                continue
            dets = [r for r in (from_detection2d(d, src["names"], src["labels"])
                                for d in msg.detections) if r is not None]
            frames.append((src["topic"], src["section"],
                           msg.header.stamp.to_sec(), dets))

        payload, warns = self.fusion.step(frames, now)
        self.out.publish(payload)

        for label, z, exp in warns:
            rospy.logwarn_throttle(
                2.0, "높이 %.2f m 가 예상 %.2f m 와 다릅니다 — "
                     "거리(pitch) 캘리브레이션을 의심하세요 [%s]", z, exp, label)

        if (self.marker_pub.get_num_connections() > 0
                and self.marker_throttle.ready()):
            self.publish_markers(payload["items"])

    # ── RViz ─────────────────────────────────────────────────────
    def publish_markers(self, items):
        arr = MarkerArray()
        for it in items:
            alpha = 0.9 if it["stable"] else 0.35
            r, gr, b = COLOR.get(it["label"], (0.8, 0.8, 0.8))
            w = max(it["width"], 0.05)
            z = it["z"]

            box = self._marker(it["id"] * 2, "obstacle")    # id 는 마커마다 고유
            box.pose.position.x = it["x"]
            box.pose.position.y = it["y"]
            if z is not None and z > 0.0:
                # 밑변 2점 + 높이 z. 깊이는 모르므로 폭과 같다고 둡니다.
                box.type = Marker.CUBE
                box.scale.x, box.scale.y, box.scale.z = w, w, z
                box.pose.position.z = z * 0.5
            else:
                box.type = Marker.CYLINDER
                box.scale.x, box.scale.y, box.scale.z = w, w, 0.05
                box.pose.position.z = 0.025
            box.color = ColorRGBA(r, gr, b, alpha)
            arr.markers.append(box)

            txt = self._marker(it["id"] * 2 + 1, "obstacle_label")
            txt.type = Marker.TEXT_VIEW_FACING
            txt.pose.position.x = it["x"]
            txt.pose.position.y = it["y"]
            txt.pose.position.z = (z or 0.0) + 0.25
            txt.scale.z = 0.2
            txt.color = ColorRGBA(1.0, 1.0, 1.0, max(alpha, 0.6))
            err = "%.1f" % it["x_err"] if it["x_err"] is not None else "?"
            txt.text = "id=%d x=%.1fm ±%s" % (it["id"], it["x"], err)
            arr.markers.append(txt)
        self.marker_pub.publish(arr)

    @staticmethod
    def _marker(mid, ns):
        m = Marker()
        m.header.frame_id = FRAME_ID          # 없으면 RViz 에 아무것도 안 뜸
        m.header.stamp = rospy.Time(0)        # Fixed Frame 이 map 이어도 최신 TF 로 그림
        m.ns = ns
        m.id = mid
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0            # 없으면 RViz 경고
        m.lifetime = rospy.Duration(0.5)      # 없으면 물체를 치워도 유령 마커가 남음
        return m

    def watchdog(self, _evt):
        for s in self.sources:
            if s["n"] == 0:
                rospy.logwarn("%s 가 아직 한 번도 안 왔습니다 — 나머지 소스로 계속 발행합니다",
                              s["topic"])


if __name__ == "__main__":
    try:
        GroundMarkers()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
