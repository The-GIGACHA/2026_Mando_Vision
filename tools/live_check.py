#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_check.py (v2) — 학습된 모델 여러 개를 한 화면에서 동시에 확인합니다.

v1 대비 바뀐 점
  1) --model 을 여러 개 줄 수 있습니다 (모델마다 박스 색이 다름)
  2) 입력이 mp4/mov 등 동영상 파일도 됩니다  (v1 은 V4L2 장치만 열렸음)
  3) 재생 제어: 일시정지 / 프레임 이동 / 반복
  4) 종료 시 모델별·클래스별 검출 통계를 표로 출력합니다
  5) --harvest 로 라벨링용 원본 프레임을 자동 추출합니다

────────────────────────────────────────────────────────────────────────
사용 예

  # 핸드폰 영상 1개에 모델 3개 동시
  python3 tools/live_check.py \
      --model weights/cone.pt weights/delivery_sign.pt weights/traffic_light_20250918.pt \
      --src ~/Videos/course.mp4

  # 차선까지 같이 (느려짐 - 아래 '주의' 참고)
  python3 tools/live_check.py \
      --model weights/lane.pt weights/cone.pt --src ~/Videos/course.mp4

  # ROS 토픽에서
  python3 tools/live_check.py --model weights/cone.pt --src /cam_left/image_raw

  # v4l2 장치에서 직접
  python3 tools/live_check.py --model weights/cone.pt --src /dev/cam_left

  # 라벨링용 프레임 15장마다 1장씩 뽑기
  python3 tools/live_check.py --model weights/cone.pt --src course.mp4 --harvest 15

키
  q / ESC   종료
  space     일시정지 / 재생
  . / ,     (일시정지 중) 다음 / 이전 프레임
  Tab       설정 대상 모델 변경
  [ / ]     선택된 모델의 conf -0.05 / +0.05
  1..9      n번째 모델 표시 on/off
  s         현재 프레임 원본+오버레이 저장
  r         영상 처음으로

────────────────────────────────────────────────────────────────────────
주의

▣ 모델을 N개 얹으면 지연도 N개 합입니다. 여기 표시되는 FPS 는
  "이 조합이 실차에서 이 속도로 돈다"는 뜻이 아닙니다.
  실제 성능 측정은 반드시 모델 1개씩 따로 하세요.

▣ lane.pt (YOLOPv2) 는 내부적으로 1280x720 로 리사이즈 후 추론합니다.
  세로 영상(9:16)을 넣으면 화면이 눌려서 결과가 무의미합니다.
  차선 확인은 반드시 가로로 찍은, 카메라를 차에 장착한 상태의 영상으로 하세요.

▣ 핸드폰 영상은 자동 노출/HDR/화이트밸런스가 걸려 있습니다.
  "무엇이 검출되는가"는 볼 수 있지만
  콘 색(blue/yellow) 판정이나 신호등 색 판정의 신뢰도는 실제 카메라와 다릅니다.
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import cv2

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm")

# 모델별 박스 색 (BGR)
MODEL_COLORS = [(90, 230, 100), (255, 190, 90), (70, 140, 255),
                (230, 100, 255), (70, 255, 255), (160, 160, 255)]


def clean_name(n):
    """'0_green_left' -> 'green_left'"""
    return re.sub(r"^\d+\s*[_\-.]\s*", "", str(n)).strip()


# ═══════════════════════════════════════════════════════════════════════
#  모델 래퍼
# ═══════════════════════════════════════════════════════════════════════
class Model(object):
    def __init__(self, path, idx, conf, utils_dir, imgsz=640):
        import torch
        self.torch = torch
        self.path = path
        self.label = os.path.splitext(os.path.basename(path))[0]
        self.color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        self.conf = conf
        # ★ 학습할 때 쓴 imgsz 와 반드시 같아야 합니다.
        #   Ultralytics predict 기본값은 640 이라, 960 으로 학습한 모델을
        #   그냥 돌리면 작은 객체가 조용히 안 잡힙니다. 에러가 안 나서
        #   "모델이 나쁘다"고 오판하기 쉬운 지점입니다.
        self.imgsz = int(imgsz)
        self.on = True
        self.last = "-"
        self.ms = None
        self.n_frames = 0        # 추론한 프레임 수
        self.n_hit = 0           # 검출이 1개 이상 있던 프레임 수
        self.cls = {}            # name -> [검출수, 최대conf]

        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dev == "cpu":
            print("!! CUDA 를 못 씁니다. CPU 로 돕니다 (YOLOPv2 는 1~2 FPS)")
        self.kind = None

        # 1) TorchScript (YOLOPv2) 먼저 시도
        try:
            m = torch.jit.load(path, map_location=self.dev).eval()
            self.half = (self.dev == "cuda")
            if self.half:
                m.half()
            if utils_dir and utils_dir not in sys.path:
                sys.path.insert(0, utils_dir)
            from utils import letterbox, driving_area_mask, lane_line_mask
            self._lb, self._da, self._ll = letterbox, driving_area_mask, lane_line_mask
            with torch.no_grad():
                z = torch.zeros(1, 3, 384, 640).to(self.dev)
                m(z.half() if self.half else z)
            self.net = m
            self.kind = "yolopv2"
            print("[%s] TorchScript/YOLOPv2   device=%s half=%s"
                  % (self.label, self.dev, self.half))
            return
        except Exception:
            pass

        # 2) Ultralytics
        from ultralytics import YOLO
        m = YOLO(path)
        m.to(self.dev)
        self.net = m
        self.kind = "ultralytics"
        self.names = {int(k): clean_name(v) for k, v in dict(m.names).items()}
        print("[%s] Ultralytics   device=%s  imgsz=%d   classes=%s"
              % (self.label, self.dev, self.imgsz, self.names))

    # ── 추론 ────────────────────────────────────────────────────────
    def run(self, im, vis):
        t0 = time.time()
        if self.kind == "yolopv2":
            info = self._run_lane(im, vis)
        else:
            info = self._run_yolo(im, vis)
        ms = (time.time() - t0) * 1000.0
        self.ms = ms if self.ms is None else 0.9 * self.ms + 0.1 * ms
        self.n_frames += 1
        return info

    def _run_yolo(self, im, vis):
        r = self.net.predict(im, conf=self.conf, imgsz=self.imgsz,
                             device=self.dev, verbose=False)[0]
        counts = {}
        if r.boxes is not None and len(r.boxes):
            self.n_hit += 1
            for b in r.boxes:
                cid = int(b.cls[0])
                sc = float(b.conf[0])
                name = self.names.get(cid, str(cid))
                counts[name] = counts.get(name, 0) + 1
                rec = self.cls.setdefault(name, [0, 0.0])
                rec[0] += 1
                rec[1] = max(rec[1], sc)
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), self.color, 2)
                txt = "%s %.2f" % (name, sc)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x1, max(0, y1 - th - 6)),
                              (x1 + tw + 4, y1), self.color, -1)
                cv2.putText(vis, txt, (x1 + 2, max(th, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return " ".join("%s=%d" % kv for kv in sorted(counts.items())) or "-"

    def _run_lane(self, im, vis):
        torch = self.torch
        im0 = cv2.resize(im, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img, _, _ = self._lb(im0, 640, stride=32)
        if img.shape[0] != 384 or img.shape[1] != 640:
            return "letterbox %dx%d (384x640 아님)" % (img.shape[1], img.shape[0])

        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        t = torch.from_numpy(x).to(self.dev)
        t = (t.half() if self.half else t.float()) / 255.0
        with torch.no_grad():
            _, seg, ll = self.net(t.unsqueeze(0))

        da = self._da(seg)
        lm = self._ll(ll)
        h, w = vis.shape[:2]
        if (da.shape[0], da.shape[1]) != (h, w):
            da = cv2.resize(da.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            lm = cv2.resize(lm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        overlay = np.zeros_like(vis)
        overlay[da > 0] = (0, 220, 0)      # 주행가능영역
        overlay[lm > 0] = (0, 0, 255)      # 차선
        m = overlay.any(axis=2)
        if m.any():
            vis[m] = (vis[m] * 0.62 + overlay[m] * 0.38).astype(np.uint8)

        lane_pct = float((lm > 0).sum()) / lm.size * 100.0
        drv_pct = float((da > 0).sum()) / da.size * 100.0
        if lane_pct > 0.05:
            self.n_hit += 1
        rec = self.cls.setdefault("lane_px%", [0, 0.0])
        rec[0] += 1
        rec[1] = max(rec[1], lane_pct)
        return "lane %.2f%%  drivable %.1f%%" % (lane_pct, drv_pct)


# ═══════════════════════════════════════════════════════════════════════
#  입력 소스
# ═══════════════════════════════════════════════════════════════════════
class FileSource(object):
    seekable = True

    def __init__(self, path, size):
        self.cap = cv2.VideoCapture(path)          # ★ V4L2 백엔드 지정 안 함
        if not self.cap.isOpened():
            sys.exit("영상을 열 수 없습니다: %s\n"
                     "  ffmpeg -i in.mov -c:v libx264 -pix_fmt yuv420p out.mp4\n"
                     "  로 변환한 뒤 다시 시도하세요." % path)
        try:                                        # 세로영상 자동 회전 (OpenCV>=4.5.2)
            self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        except Exception:
            pass
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.idx = 0
        print("영상: %s   %dx%d  %.1f fps  %d frames (%.0f초)"
              % (os.path.basename(path), w, h, self.fps, self.total,
                 self.total / max(self.fps, 1)))
        if h > w:
            print("!! 세로 영상입니다. lane.pt 결과는 신뢰하지 마세요.")

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            return None
        self.idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return f

    def seek(self, n):
        n = max(0, min(n, max(self.total - 1, 0)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        self.idx = n

    def pos(self):
        return "%d/%d" % (self.idx, self.total)

    def ok(self):
        return True


class DevSource(object):
    seekable = False

    def __init__(self, dev, size):
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            sys.exit("장치를 열 수 없습니다: %s" % dev)
        self.n = 0
        print("장치: %s" % dev)

    def read(self):
        ok, f = self.cap.read()
        if ok:
            self.n += 1
        return f if ok else None

    def pos(self):
        return "#%d" % self.n

    def ok(self):
        return True


class RosSource(object):
    seekable = False

    def __init__(self, topic, size):
        import rospy
        import threading
        from sensor_msgs.msg import Image, CompressedImage
        from cv_bridge import CvBridge
        self.rospy = rospy
        self.lock = threading.Lock()
        self.img = None
        self.n = 0
        self.bridge = CvBridge()
        rospy.init_node("live_check", anonymous=True, disable_signals=True)
        if topic.endswith("/compressed"):
            rospy.Subscriber(topic, CompressedImage, self._cb_c,
                             queue_size=1, buff_size=2 ** 24)
        else:
            rospy.Subscriber(topic, Image, self._cb,
                             queue_size=1, buff_size=2 ** 24)
        print("구독: %s" % topic)

    def _cb(self, m):
        try:
            with self.lock:
                self.img = self.bridge.imgmsg_to_cv2(m, "bgr8")
        except Exception:
            pass

    def _cb_c(self, m):
        img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            with self.lock:
                self.img = img

    def read(self):
        with self.lock:
            if self.img is None:
                return None
            self.n += 1
            return self.img.copy()

    def pos(self):
        return "#%d" % self.n

    def ok(self):
        return not self.rospy.is_shutdown()


def open_source(spec, size):
    if spec.startswith("/dev/"):
        return DevSource(spec, size)
    if os.path.isfile(spec) or spec.lower().endswith(VIDEO_EXT):
        return FileSource(spec, size)
    return RosSource(spec, size)


# ═══════════════════════════════════════════════════════════════════════
def draw_panel(vis, models, sel, src, ms_ema, paused, harvested):
    h, w = vis.shape[:2]
    rows = len(models) + 1
    ph = 20 * rows + 8
    cv2.rectangle(vis, (0, 0), (w, ph), (0, 0, 0), -1)

    head = "%.0f ms (%.1f fps)   %s" % (
        ms_ema, 1000.0 / max(ms_ema, 1e-3), src.pos())
    if paused:
        head += "   [일시정지]"
    if harvested:
        head += "   harvest=%d" % harvested
    cv2.putText(vis, head, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (200, 200, 200), 1, cv2.LINE_AA)

    for i, m in enumerate(models):
        y = 17 + 20 * (i + 1)
        mark = ">" if i == sel else " "
        state = "" if m.on else "  (off)"
        txt = "%s%d %-22s conf=%.2f  %5.0fms  %s%s" % (
            mark, i + 1, m.label, m.conf, m.ms or 0, m.last, state)
        col = m.color if m.on else (110, 110, 110)
        cv2.putText(vis, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col, 1, cv2.LINE_AA)


def summary(models):
    print("\n" + "=" * 74)
    print("모델별 검출 요약")
    print("=" * 74)
    for m in models:
        rate = 100.0 * m.n_hit / max(m.n_frames, 1)
        print("\n■ %s      프레임 검출률 %.1f%%  (%d/%d)   평균 %.0f ms"
              % (m.label, rate, m.n_hit, m.n_frames, m.ms or 0))
        if not m.cls:
            print("    검출 없음  <- conf 를 낮춰 다시 보거나, 이 코스에서 못 쓰는 모델입니다")
            continue
        for name, (cnt, mx) in sorted(m.cls.items(), key=lambda x: -x[1][0]):
            print("    %-18s %6d 회   최대 %.2f" % (name, cnt, mx))
    print("\n" + "=" * 74)
    print("판정 기준")
    print("  검출률 <5%    -> 이 코스에서 사실상 안 잡힘. 재학습 필요")
    print("  최대conf<0.5  -> 잡긴 잡으나 신뢰 못 함. 재학습 필요")
    print("  엉뚱한 클래스가 다수 -> 클래스 혼동. 라벨 재확인 + 재학습")
    print("=" * 74)


# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True,
                    help="모델 파일 여러 개 가능")
    ap.add_argument("--src", default=None,
                    help="영상파일 / /dev/xxx / ROS 토픽")
    ap.add_argument("--topic", default=None, help="(구버전 호환)")
    ap.add_argument("--device", default=None, help="(구버전 호환)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="★ 학습할 때 쓴 값과 같게. obstacle 은 960")
    ap.add_argument("--size", nargs=2, type=int, default=[1280, 720],
                    help="처리 해상도 (기본 1280 720)")
    ap.add_argument("--stride", type=int, default=1,
                    help="N 프레임마다 1장만 추론 (영상 빨리 훑을 때)")
    ap.add_argument("--harvest", type=int, default=0,
                    help="N 프레임마다 원본 1장을 라벨링용으로 저장")
    ap.add_argument("--utils", default=None, help="YOLOPv2 utils 폴더")
    ap.add_argument("--outdir", default="live_check_shots")
    ap.add_argument("--save-video", dest="save_video", default=None,
                    help="오버레이 영상 저장 경로 (예: out.mp4)")
    a = ap.parse_args()

    spec = a.src or a.topic or a.device
    if not spec:
        sys.exit("--src 가 필요합니다 (영상파일 / /dev/xxx / ROS 토픽)")

    for p in a.model:
        if not os.path.exists(p):
            sys.exit("모델이 없습니다: %s" % p)

    utils_dir = a.utils or os.path.join(
        os.path.dirname(os.path.abspath(a.model[0])), "..", "utils")
    utils_dir = os.path.abspath(utils_dir)

    models = [Model(p, i, a.conf, utils_dir, a.imgsz)
              for i, p in enumerate(a.model)]
    if len(models) > 1:
        print("\n!! 모델 %d개 동시 실행 - 여기 표시되는 FPS 는 성능 지표가 아닙니다."
              % len(models))

    src = open_source(spec, a.size)
    os.makedirs(a.outdir, exist_ok=True)
    hv_dir = os.path.join(a.outdir, "harvest")
    if a.harvest:
        os.makedirs(hv_dir, exist_ok=True)

    win = ("live_check   space:정지  ,/.:프레임  Tab:선택  [/]:conf  "
           "1-9:on/off  s:저장  q:종료")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, a.size[0], a.size[1])

    writer = None
    sel = 0
    paused = False
    ema = None
    n_saved = 0
    n_harv = 0
    raw_frame = None
    vis_frame = None
    fno = 0

    print("\n대기 중...\n")
    while src.ok():
        if not paused:
            bgr = src.read()
            if bgr is None:
                if getattr(src, "seekable", False):
                    if vis_frame is None:
                        break
                    print("영상 끝. r 로 처음부터, q 로 종료")
                    paused = True
                else:
                    if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
                        break
                    continue
            else:
                fno += 1
                if a.stride > 1 and (fno % a.stride):
                    continue

                im = cv2.resize(bgr, tuple(a.size), interpolation=cv2.INTER_LINEAR)
                raw_frame = im
                vis = im.copy()

                t0 = time.time()
                for m in models:
                    if m.on:
                        m.last = m.run(im, vis)
                dt = (time.time() - t0) * 1000.0
                ema = dt if ema is None else 0.9 * ema + 0.1 * dt

                if a.harvest and (fno % a.harvest == 0):
                    cv2.imwrite(os.path.join(hv_dir, "f%06d.png" % fno), im)
                    n_harv += 1

                draw_panel(vis, models, sel, src, ema, paused, n_harv)
                vis_frame = vis

                if a.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            a.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                            20.0, (vis.shape[1], vis.shape[0]))
                    writer.write(vis)

        if vis_frame is not None:
            show = vis_frame
            if paused:
                show = vis_frame.copy()
                draw_panel(show, models, sel, src, ema or 1.0, True, n_harv)
            cv2.imshow(win, show)

        k = cv2.waitKey(1 if not paused else 30) & 0xFF
        if k == 255:
            continue
        if k in (ord('q'), 27):
            break
        elif k == ord(' '):
            paused = not paused
        elif k == 9:                                   # Tab
            sel = (sel + 1) % len(models)
        elif k == ord('['):
            models[sel].conf = max(0.05, models[sel].conf - 0.05)
        elif k == ord(']'):
            models[sel].conf = min(0.95, models[sel].conf + 0.05)
        elif ord('1') <= k <= ord('9'):
            i = k - ord('1')
            if i < len(models):
                models[i].on = not models[i].on
        elif k == ord('s') and raw_frame is not None:
            ts = time.strftime("%H%M%S")
            cv2.imwrite(os.path.join(a.outdir, "%s_raw.png" % ts), raw_frame)
            cv2.imwrite(os.path.join(a.outdir, "%s_vis.png" % ts), vis_frame)
            n_saved += 1
            print("저장 %d: %s/%s_{raw,vis}.png" % (n_saved, a.outdir, ts))
        elif k == ord('r') and getattr(src, "seekable", False):
            src.seek(0)
            fno = 0
            paused = False
        elif k in (ord('.'), ord(',')) and getattr(src, "seekable", False):
            step = 1 if k == ord('.') else -2
            src.seek(src.idx + step)
            b = src.read()
            if b is not None:
                im = cv2.resize(b, tuple(a.size), interpolation=cv2.INTER_LINEAR)
                raw_frame = im
                vis = im.copy()
                for m in models:
                    if m.on:
                        m.last = m.run(im, vis)
                draw_panel(vis, models, sel, src, ema or 1.0, True, n_harv)
                vis_frame = vis

    if writer is not None:
        writer.release()
        print("오버레이 영상 저장: %s" % a.save_video)
    cv2.destroyAllWindows()
    if a.harvest:
        print("라벨링용 원본 %d장: %s" % (n_harv, hv_dir))
    summary(models)


if __name__ == "__main__":
    main()
