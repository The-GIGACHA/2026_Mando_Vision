#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_loop.py — 모든 인지 노드가 공유하는 두 가지 안전 장치.

2025 코드에서 반복된 두 실패 패턴을 여기서 한 번에 막습니다.

  1) 콜백 안에서 직접 추론
     → 추론이 프레임 간격보다 길면 큐에 밀리고 지연이 무한 누적됩니다.
       LatestFrame 은 항상 최신 프레임만 들고 나머지는 버립니다.

  2) "검출이 있을 때만 publish"
     → 다운스트림이 마지막 값을 영원히 붙들고 있습니다.
       (작년: 10초 전 초록불로 계속 GO 를 뿌림)
       HeartbeatPublisher 는 검출이 없어도 매 주기 "없음"을 내보냅니다.
"""

import json
import threading

import rospy
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge


class LatestFrame(object):
    """최신 프레임 1장만 보관. Image / CompressedImage 모두 지원."""

    def __init__(self, topic, msg_type=None):
        self._lock = threading.Lock()
        self._msg = None
        self._bridge = CvBridge()
        self._n_recv = 0

        if msg_type is None:
            msg_type = CompressedImage if topic.endswith("/compressed") else Image
        self._compressed = (msg_type is CompressedImage)

        # ★ queue_size=1 만으로는 부족합니다. buff_size 를 프레임보다 크게 주지
        #   않으면 rospy 가 소켓 버퍼에 쌓아두고 지연이 그대로 생깁니다.
        self._sub = rospy.Subscriber(topic, msg_type, self._cb,
                                     queue_size=1, buff_size=2 ** 24)
        rospy.loginfo("구독: %s (%s)", topic, msg_type.__name__)

    def _cb(self, msg):
        with self._lock:
            self._msg = msg            # 덮어쓰기 — 오래된 프레임은 버림
            self._n_recv += 1

    def take(self):
        """(image, header) 또는 (None, None). 한 번 꺼내면 비웁니다."""
        with self._lock:
            msg, self._msg = self._msg, None
        if msg is None:
            return None, None
        try:
            if self._compressed:
                arr = np.frombuffer(msg.data, np.uint8)
                import cv2
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn_throttle(5.0, "이미지 디코드 실패: %s", e)
            return None, None
        if img is None or img.size == 0:
            return None, None
        return img, msg.header

    @property
    def received(self):
        return self._n_recv


class HeartbeatPublisher(object):
    """
    검출 유무와 무관하게 매 주기 상태를 발행합니다.
    다운스트림은 payload 의 stamp 로 신선도를 판단하면 됩니다.
    """

    def __init__(self, topic, queue_size=1):
        self._pub = rospy.Publisher(topic, String, queue_size=queue_size)
        self._topic = topic

    def publish(self, payload):
        payload = dict(payload)
        payload["stamp"] = rospy.Time.now().to_sec()
        self._pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def publish_empty(self, **extra):
        d = {"detected": False, "label": None, "conf": 0.0}
        d.update(extra)
        self.publish(d)


class Throttle(object):
    """시각화처럼 비싼 작업의 발행 빈도를 제한합니다.

    ★ _last 를 '실제로 발행한 시각' 으로 두면 요청한 hz 가 안 나옵니다.
      주기 루프는 이산적이라 항상 기준선을 조금 넘겨서 ready() 를 부르고,
      그 초과분이 매번 다음 기준선에 누적됩니다.
          15 Hz 루프(66.7 ms) + viz_hz 5 (200 ms)
            -> 200 ms 기준선을 206.7 ms 에 넘김 -> 다음 기준선 406.7 ms
            -> 400 ms 주기는 못 쓰고 466.7 ms 에 발행 -> 실측 4.3 Hz
      기준선을 period 씩만 밀어 격자에 붙여 둡니다 (2026-09-19 실측 후).
      많이 밀렸을 때는 따라잡기를 포기하고 now 로 다시 맞춥니다."""

    def __init__(self, hz):
        self._period = 1.0 / float(hz) if hz > 0 else 0.0
        self._last = 0.0

    def ready(self):
        if self._period <= 0.0:
            return False
        now = rospy.Time.now().to_sec()
        if self._last == 0.0:
            self._last = now
            return True
        if now - self._last < self._period:
            return False
        self._last += self._period
        if now - self._last >= self._period:
            # 한 주기 이상 밀렸다 (루프가 느렸거나 처음). 몰아서 내지 않습니다.
            self._last = now
        return True
