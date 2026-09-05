#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lane_detector.py — YOLOPv2 차선 / 주행가능영역 (2026)

원본: 2025-CREAM-ERP42/src/lane_detector.py  (구조는 대체로 건전했습니다)

2025 대비 바뀐 점
  1. device 기본값 cpu → 자동(cuda 있으면 cuda). 2025 는 launch 없이 직접
     실행하면 CPU 로 돌았고 YOLOPv2 CPU 는 1~2 FPS 입니다.
  2. 콜백 추론 → 최신 프레임 + 타이머.
  3. 마스크를 raw Image 로 3개 쏘던 것(약 135 MB/s)을 정리:
     - 기본은 통계 요약만 발행 (가볍고 planning 이 실제로 쓰는 것)
     - 마스크/오버레이는 구독자가 있을 때만, throttle 걸어서 압축 발행
  4. 1280x720 강제 리사이즈의 이유를 명시하고 런타임에 검증합니다.

★ 해상도 주의
  utils.py 의 driving_area_mask / lane_line_mask 는 seg[:, :, 12:372, :] 로
  크롭합니다. 이 12px 는 1280x720(16:9)을 640 letterbox 했을 때 생기는
  상하 패딩입니다:
      r = min(640/720, 640/1280) = 0.5  →  640x360  →  dh=280
      auto: 280 % 32 = 24  →  상하 12px  →  384x640
  다른 종횡비를 넣으면 마스크가 조용히 어긋납니다. 리사이즈 줄을 지우지 마세요.

발행
  /perception/lane            (String JSON)  ← 항상, 매 주기
  /perception/lane/mask/compressed           ← 구독자 있을 때만
  /perception/lane/drivable/compressed       ← 구독자 있을 때만
  /perception/lane/overlay/compressed        ← 구독자 있을 때만
"""

import os
import sys

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "utils"))
from infer_loop import LatestFrame, HeartbeatPublisher, Throttle    # noqa: E402

PROC_W, PROC_H = 1280, 720          # ★ 위 설명 참조. 바꾸지 마세요.


class LaneDetector(object):
    def __init__(self):
        rospy.init_node("lane_detector")

        import torch
        sys.path.insert(0, os.path.join(_HERE, "..", "utils"))
        from utils import lane_line_mask, driving_area_mask, letterbox
        self._torch = torch
        self._lane_mask = lane_line_mask
        self._da_mask = driving_area_mask
        self._letterbox = letterbox

        weights = rospy.get_param("~weights", "")
        if not weights or not os.path.exists(weights):
            rospy.logfatal("가중치를 찾을 수 없습니다: %r", weights)
            raise SystemExit(1)

        self.img_size = int(rospy.get_param("~img_size", 640))
        self.infer_hz = float(rospy.get_param("~infer_hz", 15.0))
        self.viz_hz = float(rospy.get_param("~viz_hz", 5.0))
        topic = rospy.get_param("~image_topic",
                                "/cam_front/color/image_raw/compressed")

        dev = rospy.get_param("~device", "auto")
        if dev == "auto":
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        if self.device.type == "cpu":
            rospy.logwarn("=" * 58)
            rospy.logwarn("YOLOPv2 를 CPU 로 실행합니다 — 1~2 FPS 수준입니다.")
            rospy.logwarn("주행에는 쓸 수 없습니다. CUDA 설치를 확인하세요.")
            rospy.logwarn("=" * 58)

        rospy.loginfo("YOLOPv2 로드: %s  (device=%s)", weights, self.device)
        self.model = torch.jit.load(weights, map_location="cpu").to(self.device)
        self.half = (self.device.type != "cpu")
        if self.half:
            self.model.half()
        self.model.eval()

        # 워밍업 — 첫 프레임에서 수백 ms 튀는 것을 방지
        with torch.no_grad():
            z = torch.zeros(1, 3, 384, 640).to(self.device)
            self.model(z.half() if self.half else z)
        rospy.loginfo("워밍업 완료")

        self.frame = LatestFrame(topic)
        self.out = HeartbeatPublisher("/perception/lane")
        self.pub_mask = rospy.Publisher(
            "/perception/lane/mask/compressed", CompressedImage, queue_size=1)
        self.pub_da = rospy.Publisher(
            "/perception/lane/drivable/compressed", CompressedImage, queue_size=1)
        self.pub_ov = rospy.Publisher(
            "/perception/lane/overlay/compressed", CompressedImage, queue_size=1)
        self.viz_throttle = Throttle(self.viz_hz)

        self.ema_ms = None
        rospy.Timer(rospy.Duration(1.0 / self.infer_hz), self.step)
        rospy.loginfo("lane_detector 준비 완료 (%.0f Hz)", self.infer_hz)

    # ── 전처리 ───────────────────────────────────────────────────────
    def preprocess(self, bgr):
        im0 = cv2.resize(bgr, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)
        img, _, _ = self._letterbox(im0, self.img_size, stride=32)
        if img.shape[0] != 384 or img.shape[1] != 640:
            rospy.logerr_throttle(
                10.0,
                "letterbox 결과가 384x640 이 아닙니다 (%dx%d). "
                "utils.py 의 12:372 크롭 전제가 깨집니다.",
                img.shape[1], img.shape[0])
            return None, None
        x = img[:, :, ::-1].transpose(2, 0, 1)          # BGR→RGB, HWC→CHW
        x = np.ascontiguousarray(x)
        t = self._torch.from_numpy(x).to(self.device)
        t = t.half() if self.half else t.float()
        t /= 255.0
        return im0, t.unsqueeze(0)

    # ── 메인 루프 ────────────────────────────────────────────────────
    def step(self, _evt):
        bgr, _hdr = self.frame.take()
        if bgr is None:
            self.out.publish({"detected": False, "lane_ratio": 0.0,
                              "drivable_ratio": 0.0, "infer_ms": None})
            return

        im0, tensor = self.preprocess(bgr)
        if tensor is None:
            self.out.publish({"detected": False, "lane_ratio": 0.0,
                              "drivable_ratio": 0.0, "infer_ms": None})
            return

        t0 = rospy.Time.now().to_sec()
        with self._torch.no_grad():
            # ★ pred 는 텐서가 아니라 3개 텐서의 '리스트'입니다.
            #   객체 검출까지 쓰려면 utils.split_for_trace_model 에 넘기세요.
            [pred, anchor_grid], seg, ll = self.model(tensor)
        infer_ms = (rospy.Time.now().to_sec() - t0) * 1000.0
        self.ema_ms = infer_ms if self.ema_ms is None else \
            0.9 * self.ema_ms + 0.1 * infer_ms

        da = self._da_mask(seg)          # (720, 1280) 0/1
        ll_m = self._lane_mask(ll)       # (720, 1280) 0/1

        total = float(da.size)
        lane_ratio = float((ll_m > 0).sum()) / total
        da_ratio = float((da > 0).sum()) / total

        self.out.publish({
            "detected": lane_ratio > 0.0005,     # 차선 픽셀이 유의미하게 있는가
            "lane_ratio": round(lane_ratio, 5),
            "drivable_ratio": round(da_ratio, 5),
            "infer_ms": round(self.ema_ms, 1),
            "shape": [int(da.shape[0]), int(da.shape[1])],
        })

        if self.viz_throttle.ready():
            self.publish_viz(im0, da, ll_m)

    # ── 시각화 (구독자 있을 때만) ────────────────────────────────────
    def publish_viz(self, im0, da, ll_m):
        if self.pub_mask.get_num_connections() > 0:
            self._send(self.pub_mask, (ll_m > 0).astype(np.uint8) * 255, png=True)
        if self.pub_da.get_num_connections() > 0:
            self._send(self.pub_da, (da > 0).astype(np.uint8) * 255, png=True)
        if self.pub_ov.get_num_connections() > 0:
            ov = im0.copy()
            color = np.zeros_like(ov)
            color[da > 0] = (0, 255, 0)        # 주행가능영역 — 초록
            color[ll_m > 0] = (0, 0, 255)      # 차선 — 빨강
            m = color.any(axis=2)
            ov[m] = (ov[m] * 0.5 + color[m] * 0.5).astype(np.uint8)
            self._send(self.pub_ov, ov, png=False)

    @staticmethod
    def _send(pub, img, png):
        if png:
            ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            fmt = "png"
        else:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            fmt = "jpeg"
        if not ok:
            return
        m = CompressedImage()
        m.header.stamp = rospy.Time.now()
        m.format = fmt
        m.data = buf.tobytes()
        pub.publish(m)


if __name__ == "__main__":
    try:
        LaneDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
