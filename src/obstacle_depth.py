#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obstacle_depth.py — S자 코스에서만 bbox + 정렬 뎁스 -> /obstacles/camera (base_link 점)

  obstacle_detector ─ /detect/obstacle ─┐
  cone_detector     ─ /detect/cone     ─┼─ obstacle_depth ─ /obstacles/camera          PointCloud2  ← 제어 (S코스)
  D455 aligned_depth_to_color          ─┘                 └ /perception/obstacle_depth  String JSON  ← bev_viz, 확인용

★ 왜 S자 코스만인가 (2026-09-17 사용자 결정)
  S코스는 장애물 바로 옆을 비켜 지나가서 가까운 거리(밑동이 화면 밖으로 잘리는 2.70 m 안)가
  판정에 들어온다. 지면 투영만으로는 그 거리를 못 잰다. 다른 구간은 /perception/obstacle x/y 로 충분하다.
  제어도 /obstacles/camera 를 S_COURSE 에서만 쓴다.

★ 뎁스를 그대로 믿지 않는다. 박스마다 지면 투영과 대조해 맞을 때만 뎁스 점을 내고,
  아니면 지면 투영 점을 낸다 (규칙: utils/obstacle_depth.py). S코스 좌/우 Bool 은 여기와 무관하게
  지면 투영으로 판정한다 — 역광 뎁스 오인식이 한 번 커밋되면 안 뒤집히는 판정에 들어가면 안 된다.

★ 카메라 기하는 ground.py 한 벌 (~ground/*). 시동 때 TF(URDF) 와 대조하고,
  주행 중에는 노면 뎁스 평면으로 pitch 를 재서 ~ground/cam_pitch_deg 와 다르면 경고한다.

발행 규칙 (docs/OBSTACLE_CAMERA_TOPIC.md)
  · header.stamp = 영상 캡처 시각, frame_id = base_link
  · 영상을 처리했으면 장애물이 없어도 빈 cloud 를 낸다 ('봤는데 없음')
  · 미션 게이트가 꺼져 있거나 검출 영상이 안 오면 내지 않는다 ('모름')
"""
import collections
import math
import os
import sys
import threading

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection2DArray


def _add_utils_to_path():
    """catkin_install_python 이 스크립트를 devel/lib/<pkg>/ 로 복사하므로
    __file__ 기준 '../utils' 가 거기서는 존재하지 않습니다."""
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "..", "utils")
    if not os.path.isdir(cand):
        import rospkg
        cand = os.path.join(rospkg.RosPack().get_path("mando_vision_2026"), "utils")
    if cand not in sys.path:
        sys.path.insert(0, cand)


_add_utils_to_path()

import class_map                                                          # noqa: E402
import mission_gate                                                       # noqa: E402
from ground import from_params                                            # noqa: E402
from infer_loop import HeartbeatPublisher, Throttle                       # noqa: E402
from obstacle_depth import (DepthProjector, decode_compressed_depth,      # noqa: E402
                            fit_ground_plane)
from obstacle_ground import FRAME_ID, from_detection2d                    # noqa: E402

DEFAULT_SOURCES = [
    {"topic": "/detect/obstacle", "section": "obstacle", "labels": ["T870"]},
]


class ObstacleDepth(object):
    def __init__(self):
        rospy.init_node("obstacle_depth")
        g = rospy.get_param

        self.G = from_params(g)
        self.DP = DepthProjector(
            self.G,
            wheel_radius=g("~wheel_radius", 0.150),
            cols=g("~depth/cols", 5),
            percentile=g("~depth/percentile", 30.0),
            min_valid=g("~depth/min_valid", 0.2),
            min_cols=g("~depth/min_cols", 3),
            depth_min_m=g("~depth/min_m", 0.3),
            depth_max_m=g("~depth/max_m", 8.0),
            tol_abs_m=g("~depth/tol_abs_m", 0.30),
            tol_rel=g("~depth/tol_rel", 0.15),
            pitch_tol_deg=g("~ground/pitch_tol_deg", 0.5),
            image_width=g("~image_width", 1280),
            image_height=g("~image_height", 720))
        self.sync_sec = float(g("~depth/sync_sec", 0.05))
        self.source_timeout = float(g("~source_timeout", 0.5))
        self.rate_hz = float(g("~rate_hz", 15.0))

        self._lock = threading.Lock()
        self.sources = [self._make_source(i, s)
                        for i, s in enumerate(g("~sources", DEFAULT_SOURCES))]

        # 뎁스는 30 Hz, 검출은 10~15 Hz 에 추론 지연까지 있어 검출 영상 stamp 에 맞는 프레임을 1초치 들고 고른다
        self.depth_buf = collections.deque(maxlen=int(g("~depth/buffer", 30)))
        self._decoded = (None, None)                  # (stamp, 배열 m)
        depth_topic = g("~depth_topic", "/cam_front/aligned_depth_to_color/image_raw")
        self._depth_compressed = depth_topic.endswith("compressedDepth")
        rospy.Subscriber(depth_topic, CompressedImage if self._depth_compressed else Image,
                         self._cb_depth, queue_size=2, buff_size=2 ** 24)

        self.gate_enable = bool(g("~gate/enable", True))
        self.gate = mission_gate.MissionGate(g("~gate/missions", {"S_COURSE": 0.0}),
                                             stale_sec=g("~gate/stale_sec", 2.0))
        if self.gate_enable:
            mission_gate.attach(self.gate, rospy)
        self._gate_mode = None

        self.cloud_pub = rospy.Publisher(g("~output_topic", "/obstacles/camera"),
                                         PointCloud2, queue_size=1)
        self.debug = HeartbeatPublisher(g("~debug_topic", "/perception/obstacle_depth"))
        self.idle_throttle = Throttle(2.0)

        # 노면 평면으로 pitch 감시. 뎁스와 지면 투영 대조는 pitch 가 맞다는 전제 위에 서 있다.
        self.plane_throttle = Throttle(float(g("~plane/hz", 1.0)))
        self.plane_warn_deg = float(g("~plane/warn_deg", 1.0))
        self.plane_hist = collections.deque(maxlen=int(g("~plane/window", 10)))
        self.plane = None

        for i, src in enumerate(self.sources):
            rospy.Subscriber(src["topic"], Detection2DArray, self._cb_det,
                             callback_args=i, queue_size=1)

        self.camera_frame = g("~camera_frame", "cam_front_color_optical_frame")
        if g("~tf_check", True):
            rospy.Timer(rospy.Duration(3.0), self._tf_check, oneshot=True)
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.step)
        rospy.loginfo("obstacle_depth 준비 (뎁스 %s, 게이트 %s, h=%.3f pitch=%.2f°) 소스: %s",
                      depth_topic, dict(self.gate.missions) if self.gate_enable else "off",
                      self.G.h, math.degrees(self.G.th),
                      ", ".join("%s[%s]" % (s["topic"], "/".join(sorted(s["labels"])))
                                for s in self.sources))

    def _make_source(self, i, s):
        try:
            topic, section = str(s["topic"]), str(s["section"])
            labels = [str(v) for v in s["labels"]]
        except (KeyError, TypeError):
            rospy.logfatal("~sources[%d] 에 topic / section / labels 가 모두 필요합니다: %r", i, s)
            raise SystemExit(1)
        names = class_map.load(section)["names"]
        unknown = sorted(set(labels) - set(names.values()))
        if not labels or unknown:
            # 오타 난 라벨은 에러 없이 전부 걸러져 '장애물 없음' 으로 나간다
            rospy.logfatal("~sources[%d] %s: labels %s 가 classes.yaml '%s' 에 없습니다",
                           i, topic, unknown or "(비어 있음)", section)
            raise SystemExit(1)
        return {"topic": topic, "labels": set(labels), "names": names,
                "msg": None, "res": None}

    # ── 입력 ─────────────────────────────────────────────────────
    def _cb_depth(self, msg):
        with self._lock:
            self.depth_buf.append((msg.header.stamp.to_sec(), msg))

    def _cb_det(self, msg, i):
        with self._lock:
            self.sources[i]["msg"] = msg

    def _decode(self, msg):
        if self._depth_compressed:
            raw = decode_compressed_depth(msg.data)
            return None if raw is None else raw.astype(np.float32) * 0.001
        if msg.encoding == "16UC1":
            dt = np.dtype(">u2" if msg.is_bigendian else "<u2")
            arr = np.frombuffer(msg.data, dt).reshape(msg.height, msg.step // 2)[:, :msg.width]
            return arr.astype(np.float32) * 0.001
        if msg.encoding == "32FC1":
            dt = np.dtype(">f4" if msg.is_bigendian else "<f4")
            return np.frombuffer(msg.data, dt).reshape(msg.height, msg.step // 4)[:, :msg.width].copy()
        rospy.logwarn_throttle(10.0, "뎁스 인코딩 %s 는 지원 안 함 — 지면 투영만 씁니다", msg.encoding)
        return None

    def depth_for(self, stamp):
        """검출 영상 stamp 에 가장 가까운 뎁스 (m 배열). sync_sec 밖이면 None."""
        with self._lock:
            buf = list(self.depth_buf)
        if not buf:
            return None
        st, msg = min(buf, key=lambda p: abs(p[0] - stamp))
        if abs(st - stamp) > self.sync_sec:
            return None
        if self._decoded[0] != st:
            self._decoded = (st, self._decode(msg))
        return self._decoded[1]

    # ── 주기 ─────────────────────────────────────────────────────
    def _gate(self, now):
        if not self.gate_enable:
            return True, {"mode": "disabled"}
        active, info = self.gate.check(now)
        if info["mode"] != self._gate_mode:
            rospy.loginfo("게이트 %s -> %s (%s)", self._gate_mode, info["mode"], info.get("mission"))
            self._gate_mode = info["mode"]
        if info["mode"] in ("no_grm", "no_map"):
            rospy.logwarn_throttle(10.0, "GRM 구간 정보 없음(%s) — 항상 켜고 /obstacles/camera 를 냅니다",
                                   info["mode"])
        return active, info

    def step(self, _evt):
        now = rospy.Time.now().to_sec()
        active, ginfo = self._gate(now)
        if not active:
            for s in self.sources:
                s["msg"], s["res"] = None, None
            if self.idle_throttle.ready():
                self.debug.publish({"active": False, "gate": ginfo, "items": [],
                                    "image_stamp": None, "plane": self.plane})
            return

        new = False
        for s in self.sources:
            with self._lock:
                msg, s["msg"] = s["msg"], None
            if msg is None or msg.header.stamp.is_zero():
                continue                       # stamp 0 = 검출 노드가 영상을 못 받음
            st = msg.header.stamp.to_sec()
            depth = self.depth_for(st)
            items, pts = [], []
            for det in msg.detections:
                r = from_detection2d(det, s["names"], s["labels"])
                if r is None:
                    continue
                label, conf, x1, y1, x2, y2 = r
                m = self.DP.measure((x1, y1, x2, y2), depth)
                pts.extend(m.pop("points"))
                m.update(label=label, conf=round(conf, 3))
                items.append(m)
            s["res"] = (now, st, items, pts, depth is not None)
            new = True
            if depth is not None and self.plane_throttle.ready():
                self._plane(depth)
        if not new:
            return

        fresh = [s["res"] for s in self.sources
                 if s["res"] is not None and now - s["res"][0] <= self.source_timeout]
        stamp = max(r[1] for r in fresh)
        pts = [p for r in fresh for p in r[3]]
        header = Header(stamp=rospy.Time.from_sec(stamp), frame_id=FRAME_ID)
        self.cloud_pub.publish(pc2.create_cloud_xyz32(header, pts))

        items = sorted((it for r in fresh for it in r[2]),
                       key=lambda it: (it["ground"] or {}).get("x", 99.0))
        if not all(r[4] for r in fresh):
            rospy.logwarn_throttle(5.0, "검출 영상 시각에 맞는 뎁스가 없습니다 — 지면 투영으로 대신합니다")
        self.debug.publish({"active": True, "gate": ginfo, "frame_id": FRAME_ID,
                            "image_stamp": stamp, "age_ms": round((now - stamp) * 1000.0, 1),
                            "n_points": len(pts), "items": items, "plane": self.plane,
                            "ground_pitch_deg": round(math.degrees(self.G.th), 2)})

    # ── 기하 감시 ────────────────────────────────────────────────
    def _plane(self, depth):
        G = self.G
        f = fit_ground_plane(depth, G.fx, G.fy, G.cx, G.cy)
        if f is None or f["inlier"] < 0.5:
            return
        self.plane_hist.append(f)
        med = {k: round(float(np.median([p[k] for p in self.plane_hist])), 3)
               for k in ("h", "pitch_deg", "roll_deg")}
        med["n"] = len(self.plane_hist)
        self.plane = med
        diff = med["pitch_deg"] - math.degrees(G.th)
        if med["n"] >= 5 and abs(diff) > self.plane_warn_deg:
            rospy.logwarn_throttle(
                10.0, "노면 뎁스 평면 pitch %.2f° 가 ~ground/cam_pitch_deg %.2f° 와 %.1f° 다릅니다 "
                      "(높이 %.3f m) — 지면 투영 거리가 짧게 나와 대조 mismatch 가 늘어납니다",
                med["pitch_deg"], math.degrees(G.th), diff, med["h"])

    def _tf_check(self, _evt):
        """URDF(TF) 와 ~ground/* 가 같은 카메라를 말하는지. 다르면 에러 없이 좌표만 틀어진다."""
        try:
            import tf2_ros
            buf = tf2_ros.Buffer()
            tf2_ros.TransformListener(buf)
            tr = buf.lookup_transform(FRAME_ID, self.camera_frame, rospy.Time(0),
                                      rospy.Duration(3.0)).transform
        except Exception as e:                                     # noqa: BLE001
            rospy.logwarn("TF %s -> %s 조회 실패 — URDF 대조를 건너뜁니다 (%s)",
                          FRAME_ID, self.camera_frame, e)
            return
        t, q = tr.translation, tr.rotation
        # 광축(광학 z) 을 base_link 로 돌린다
        fx = 2 * (q.x * q.z + q.w * q.y)
        fy = 2 * (q.y * q.z - q.w * q.x)
        fz = 1 - 2 * (q.x * q.x + q.y * q.y)
        pitch = math.degrees(math.atan2(-fz, math.hypot(fx, fy)))
        yaw = math.degrees(math.atan2(fy, fx))
        G = self.G
        tf_h = t.z + self.DP.wheel_radius
        bad = []
        if abs(t.x - G.x_off) > 0.02 or abs(t.y - G.y_off) > 0.02 or abs(tf_h - G.h) > 0.02:
            bad.append("위치 TF (%.3f, %.3f, 높이 %.3f) vs ground (%.3f, %.3f, %.3f)"
                       % (t.x, t.y, tf_h, G.x_off, G.y_off, G.h))
        if abs(pitch - math.degrees(G.th)) > 0.3:
            bad.append("pitch TF %.2f° vs ground %.2f°" % (pitch, math.degrees(G.th)))
        if abs(yaw) > 0.3:
            bad.append("yaw TF %.2f° (ground.py 는 yaw 0 가정)" % yaw)
        if bad:
            rospy.logwarn("URDF 와 ~ground/* 가 다릅니다: %s", "; ".join(bad))
        else:
            rospy.loginfo("URDF 대조 일치: 위치 (%.3f, %.3f, 높이 %.3f), pitch %.2f°",
                          t.x, t.y, tf_h, pitch)


if __name__ == "__main__":
    try:
        ObstacleDepth()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
