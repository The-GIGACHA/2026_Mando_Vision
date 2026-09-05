#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stopline_detector.py — 정지선 검출 (OpenCV)

발행: /perception/stopline  (std_msgs/String, JSON)
  {"detected": true, "row": 512, "thickness_px": 25, "angle_deg": 4.0,
   "fill": 0.96, "distance_m": 1.85, "crosswalk": false,
   "votes": 4, "window": 5, "stamp": 1234567.89}

  row        정지선 근접 모서리의 y 좌표 (원본 이미지 기준, ROI 가로 중앙)
  angle_deg  이미지상 기울기. + = 오른쪽으로 내려감
  distance_m 호모그래피를 설정했을 때만. 아니면 null
  crosswalk  횡단보도로 판단되면 true (이때 detected 는 false)
"""

import os
import sys
from collections import deque

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "utils"))
from infer_loop import LatestFrame, HeartbeatPublisher, Throttle


class StoplineDetector(object):
    def __init__(self):
        rospy.init_node("stopline_detector")

        topic = rospy.get_param("~image_topic",
                                "/cam_stopline/image_raw/compressed")
        self.rate_hz = float(rospy.get_param("~rate_hz", 15.0))
        self.viz_hz = float(rospy.get_param("~viz_hz", 5.0))

        self.roi = rospy.get_param("~roi", [])
        if self.roi and len(self.roi) != 4:
            rospy.logfatal("~roi 는 [x1,y1,x2,y2] 4개여야 합니다: %s", self.roi)
            raise SystemExit(1)

        self.tau = int(rospy.get_param("~tau", 30))

        self.min_resp = int(rospy.get_param("~min_response", 25))

        self.open_w = int(rospy.get_param("~open_width", 3))
        self.close_w = int(rospy.get_param("~close_width", 25))

        self.max_angle = float(rospy.get_param("~max_angle_deg", 20.0))
        self.angle_step = float(rospy.get_param("~angle_step_deg", 2.0))
        n = int(self.max_angle / self.angle_step)
        self.angles = [i * self.angle_step for i in range(-n, n + 1)]

        self.min_fill = float(rospy.get_param("~min_fill", 0.55))
        self.min_thick = int(rospy.get_param("~min_thickness", 10))
        self.max_thick = int(rospy.get_param("~max_thickness", 120))

        self.crosswalk_bands = int(rospy.get_param("~crosswalk_min_bands", 3))

        self.vote_win = int(rospy.get_param("~vote_window", 5))
        self.vote_min = int(rospy.get_param("~vote_min", 3))

        self.H = self._build_homography(
            rospy.get_param("~image_points", []),
            rospy.get_param("~world_points", []))

        self.frame = LatestFrame(topic)
        self.out = HeartbeatPublisher("/perception/stopline")
        self.viz_pub = rospy.Publisher(
            "/perception/stopline/viz/compressed", CompressedImage, queue_size=1)
        self.viz_throttle = Throttle(self.viz_hz)

        self.votes = deque(maxlen=self.vote_win)
        self.last = None

        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.step)
        rospy.Timer(rospy.Duration(5.0), self.watchdog)
        rospy.loginfo("stopline_detector 준비 완료 (%.0f Hz, 각도탐색 %d개, 거리환산=%s)",
                      self.rate_hz, len(self.angles),
                      "ON" if self.H is not None else "OFF")

    def _build_homography(self, img_pts, wld_pts):
        if not img_pts or not wld_pts:
            rospy.logwarn("~image_points/~world_points 미설정 — distance_m 은 null 로 나갑니다")
            return None
        if len(img_pts) != 4 or len(wld_pts) != 4:
            rospy.logfatal("~image_points 와 ~world_points 는 각각 4점이어야 합니다")
            raise SystemExit(1)
        src = np.array(img_pts, dtype=np.float32)
        dst = np.array(wld_pts, dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        rospy.loginfo("호모그래피 설정 완료 — 거리(m)를 발행합니다")
        return H

    def _to_ground(self, u, v):
        if self.H is None:
            return None
        p = np.array([[[float(u), float(v)]]], dtype=np.float32)
        q = cv2.perspectiveTransform(p, self.H)[0][0]
        return round(float(q[0]), 3)

    def crop(self, img):
        if not self.roi:
            return img, (0, 0)
        h, w = img.shape[:2]
        x1 = max(0, min(int(self.roi[0]), w - 1))
        y1 = max(0, min(int(self.roi[1]), h - 1))
        x2 = max(x1 + 1, min(int(self.roi[2]), w))
        y2 = max(y1 + 1, min(int(self.roi[3]), h))
        return img[y1:y2, x1:x2], (x1, y1)

    def _dbd(self, gray):
        """
        Dark-Bright-Dark 필터 — 이 노드의 핵심.

            f(y) = 2*I(y) - I(y-tau) - I(y+tau) - |I(y-tau) - I(y+tau)|

        ★ 절대 밝기가 아니라 '위아래와의 차이'만 봅니다.
          → 그늘이든 역광이든 노면 대비 밝은 띠면 응답이 나옵니다.
        ★ 마지막 절대값 항이 '한쪽만 어두운' 경우를 걸러냅니다.
          (예: 노면→그림자 경계는 위아래 밝기가 비대칭이라 응답이 상쇄됨)
        """
        g = gray.astype(np.int16)
        up = np.roll(g, self.tau, axis=0)
        dn = np.roll(g, -self.tau, axis=0)
        resp = 2 * g - up - dn - np.abs(up - dn)
        resp[:self.tau, :] = 0
        resp[-self.tau:, :] = 0
        return np.clip(resp, 0, 255).astype(np.uint8)

    def _marking_mask(self, gray):
        resp = self._dbd(gray)
        th, _ = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = max(th, float(self.min_resp))
        _, m = cv2.threshold(resp, th, 255, cv2.THRESH_BINARY)
        if self.open_w > 1:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.open_w, 1))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if self.close_w > 1:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.close_w, 1))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return m, float(th)

    @staticmethod
    def _shear(mask, ang_deg):
        """
        기울기 ang_deg 인 직선이 수평이 되도록 세로로 밉니다.

        ★ 가로 중앙을 축으로 밀기 때문에, 중앙 열의 y 좌표는 변하지 않습니다.
          → 전단된 이미지의 행 번호를 그대로 원본 행으로 쓸 수 있습니다.
        """
        h, w = mask.shape
        m = np.tan(np.radians(ang_deg))
        M = np.float32([[1.0, 0.0, 0.0],
                        [-m, 1.0, m * (w / 2.0)]])
        return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _best_shear(self, mask):
        """
        여러 각도로 밀어 보고 '한 행의 최대 채움률'이 가장 큰 각도를 고릅니다.
        탐색은 절반 크기에서 (빠르게), 확정된 각도로만 원본 크기에서 다시 전단합니다.
        """
        h, w = mask.shape
        small = cv2.resize(mask, (max(w // 2, 1), max(h // 2, 1)),
                           interpolation=cv2.INTER_AREA)
        best_ang, best_peak = 0.0, -1.0
        for a in self.angles:
            s = self._shear(small, a)
            prof = s.sum(axis=1) / (255.0 * s.shape[1])
            peak = float(prof.max()) if prof.size else 0.0
            if peak > best_peak:
                best_ang, best_peak = a, peak
        return best_ang, self._shear(mask, best_ang)

    def _bands(self, mask):
        """행별 채움률이 임계 이상인 '연속된 행 구간'을 모두 찾습니다."""
        h, w = mask.shape
        fill = mask.sum(axis=1) / (255.0 * w)
        hot = fill >= self.min_fill

        bands, y = [], 0
        while y < h:
            if not hot[y]:
                y += 1
                continue
            y0 = y
            while y < h and hot[y]:
                y += 1
            y1 = y - 1
            thick = y1 - y0 + 1
            if self.min_thick <= thick <= self.max_thick:
                bands.append((y0, y1, float(fill[y0:y1 + 1].mean())))
        return bands, fill

    def step(self, _evt):
        img, _hdr = self.frame.take()
        if img is None:
            self.votes.append(False)
            self.emit(None, False)
            return

        meas, crosswalk, dbg = self.detect(img)
        self.votes.append(meas is not None)
        self.emit(meas, crosswalk)

        if self.viz_pub.get_num_connections() > 0 and self.viz_throttle.ready():
            self.publish_viz(img, meas, crosswalk, dbg)

    def detect(self, img):
        """이미지 1장 → (측정값 dict 또는 None, 횡단보도 여부, 시각화용 중간결과)"""
        roi_img, (ox, oy) = self.crop(img)
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        mask, th = self._marking_mask(gray)
        ang, sheared = self._best_shear(mask)
        bands, _fill = self._bands(sheared)

        crosswalk = len(bands) >= self.crosswalk_bands

        meas = None
        if bands and not crosswalk:
            y0, y1, f = max(bands, key=lambda b: b[1])
            near = int(y1 + oy)
            cx = int(mask.shape[1] // 2 + ox)
            meas = {
                "row": near,
                "thickness_px": int(y1 - y0 + 1),
                "angle_deg": round(float(ang), 2),
                "fill": round(f, 3),
                "distance_m": self._to_ground(cx, near),
            }
        dbg = {"mask": mask, "off": (ox, oy), "angle": ang,
               "bands": bands, "th": th}
        return meas, crosswalk, dbg

    def emit(self, meas, crosswalk):
        """N프레임 다수결로 안정화해 발행합니다. 검출이 없어도 매 주기 발행합니다."""
        n = sum(1 for v in self.votes if v)
        stable = n >= self.vote_min

        if meas is not None:
            self.last = meas

        d = {"detected": bool(stable), "crosswalk": bool(crosswalk),
             "votes": n, "window": self.vote_win,
             "row": None, "thickness_px": None, "angle_deg": None,
             "fill": None, "distance_m": None}
        if stable and self.last is not None:
            for k in ("row", "thickness_px", "angle_deg", "fill", "distance_m"):
                d[k] = self.last[k]
        self.out.publish(d)

    def publish_viz(self, img, meas, crosswalk, dbg):
        mask = dbg["mask"]
        ox, oy = dbg["off"]
        vis = img.copy()
        h, w = mask.shape

        sub = vis[oy:oy + h, ox:ox + w]
        sel = mask > 0
        sub[sel] = (sub[sel] * 0.5 + np.array((255, 255, 0)) * 0.5).astype(np.uint8)

        if self.roi:
            cv2.rectangle(vis, (int(self.roi[0]), int(self.roi[1])),
                          (int(self.roi[2]), int(self.roi[3])), (255, 0, 255), 2)

        m = np.tan(np.radians(dbg["angle"]))
        def line_at(row):
            y_l = int(round(row + m * (0 - w / 2.0))) + oy
            y_r = int(round(row + m * (w - 1 - w / 2.0))) + oy
            return (ox, y_l), (ox + w - 1, y_r)

        for y0, y1, _f in dbg["bands"]:
            p1, p2 = line_at(y0)
            q1, q2 = line_at(y1)
            cv2.line(vis, p1, p2, (128, 128, 128), 1)
            cv2.line(vis, q1, q2, (128, 128, 128), 1)

        if meas is not None:
            p1, p2 = line_at(meas["row"] - oy)
            cv2.line(vis, p1, p2, (0, 255, 0), 3)
            txt = "STOPLINE t=%dpx ang=%.1f fill=%.2f" % (
                meas["thickness_px"], meas["angle_deg"], meas["fill"])
            if meas["distance_m"] is not None:
                txt += " d=%.2fm" % meas["distance_m"]
            cv2.putText(vis, txt, (ox + 5, max(20, p1[1] - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        elif crosswalk:
            cv2.putText(vis, "CROSSWALK (%d bands) - not a stopline" % len(dbg["bands"]),
                        (ox + 5, oy + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 165, 255), 2, cv2.LINE_AA)

        cv2.putText(vis, "otsu=%.0f  shear=%.0fdeg  bands=%d"
                    % (dbg["th"], dbg["angle"], len(dbg["bands"])),
                    (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = buf.tobytes()
        self.viz_pub.publish(msg)

    def watchdog(self, _evt):
        if self.frame.received == 0:
            rospy.logwarn("정지선 카메라 프레임이 아직 한 장도 오지 않았습니다")


if __name__ == "__main__":
    try:
        StoplineDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
