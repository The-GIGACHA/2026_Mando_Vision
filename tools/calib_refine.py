#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calib_refine.py — LiDAR 포인트를 카메라 영상에 겹쳐 그리고, 트랙바로 외부
                  파라미터를 눈으로 맞추는 도구.

줄자·수평계로 잰 초기값을 URDF에 넣어두면 보통 3~8 cm / 2~4도 오차가 납니다.
이 도구는 그 오차를 실시간 오버레이를 보며 손으로 좁히는 용도입니다.
체커보드 없이도 융합에 쓸 만한 수준(1~2 px)까지 갑니다.

  ┌─ 사용 순서 ────────────────────────────────────────────────────────┐
  │ 1. URDF에 실측값을 넣고 robot_state_publisher 를 띄운다             │
  │ 2. LiDAR + 대상 카메라를 켠다                                       │
  │ 3. 이 스크립트 실행 → 트랙바로 오버레이가 맞을 때까지 조정          │
  │ 4. 's' → 보정된 xacro property 값이 출력됨 → URDF에 붙여넣기        │
  └────────────────────────────────────────────────────────────────────┘

  ★ 맞추기 좋은 장면: 벽 모서리, 기둥, 문틀, 책상 모서리처럼
    깊이가 급격히 변하는 수직/수평 경계. 평평한 벽만 보면 절대 안 맞습니다.

예:
  python3 calib_refine.py --camera-frame cam_left_optical_frame \
      --image /usb_cam_left/image_raw --info /usb_cam_left/camera_info \
      --cloud /ouster/points --urdf-prefix left

조작:
  트랙바   : dx dy dz (mm) / droll dpitch dyaw (0.1도)
  s        : 현재 값을 xacro 형식으로 출력 + YAML 저장
  r        : 트랙바 전부 0으로 리셋
  [ / ]    : 표시 점 크기 감소 / 증가
  q / ESC  : 종료
"""

import argparse
import sys

import numpy as np
import cv2

import rospy
import tf2_ros
import tf.transformations as tft
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge

WIN = "calib_refine  [s]ave  [r]eset  [q]uit"

# _link -> _optical_frame 고정 회전 (URDF의 rpy=-90,0,-90 과 동일)
T_LINK_OPT = tft.euler_matrix(-np.pi / 2, 0.0, -np.pi / 2, axes="sxyz")


def nothing(_):
    pass


class Refiner:
    def __init__(self, a):
        self.a = a
        self.bridge = CvBridge()
        self.img = None
        self.cloud = None
        self.K = None
        self.D = None
        self.pt_size = 2

        self.tf_buf = tf2_ros.Buffer(rospy.Duration(30.0))
        self.tf_lis = tf2_ros.TransformListener(self.tf_buf)

        if a.image.endswith("/compressed"):
            rospy.Subscriber(a.image, CompressedImage, self.cb_img_c, queue_size=1,
                             buff_size=2 ** 24)
        else:
            rospy.Subscriber(a.image, Image, self.cb_img, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(a.cloud, PointCloud2, self.cb_cloud, queue_size=1, buff_size=2 ** 26)

        if a.fx:
            self.K = np.array([[a.fx, 0, a.cx], [0, a.fy, a.cy], [0, 0, 1]], float)
            self.D = np.zeros(5)
            rospy.loginfo("내부 파라미터: 명령행 인자 사용")
        else:
            rospy.Subscriber(a.info, CameraInfo, self.cb_info, queue_size=1)

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1280, 720)
        # 오프셋 중앙 = 델타 0
        for n, m in (("dx  mm", 500), ("dy  mm", 500), ("dz  mm", 500),
                     ("droll  0.1deg", 150), ("dpitch 0.1deg", 150), ("dyaw   0.1deg", 150)):
            cv2.createTrackbar(n, WIN, m, 2 * m, nothing)
        cv2.createTrackbar("max range m", WIN, 20, 60, nothing)

    # ── 콜백 ──────────────────────────────────────────────────────────
    def cb_img(self, m):
        try:
            self.img = self.bridge.imgmsg_to_cv2(m, "bgr8")
        except Exception as e:
            rospy.logwarn_throttle(5, "image decode: %s" % e)

    def cb_img_c(self, m):
        arr = np.frombuffer(m.data, np.uint8)
        self.img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def cb_cloud(self, m):
        self.cloud = m

    def cb_info(self, m):
        if self.K is None:
            self.K = np.array(m.K, float).reshape(3, 3)
            self.D = np.array(m.D, float)
            rospy.loginfo("내부 파라미터 수신: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                          self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2])
            if abs(self.K[0, 0] - 1.0) < 1e-6 or self.K[0, 0] == 0.0:
                rospy.logerr("★ camera_info가 캘리브레이션되지 않았습니다. "
                             "--fx/--fy/--cx/--cy 로 근사값을 주거나 먼저 내부 캘리를 하세요.")

    # ── 트랙바 → 델타 변환행렬 ────────────────────────────────────────
    def read_delta(self):
        g = lambda n: cv2.getTrackbarPos(n, WIN)
        dx = (g("dx  mm") - 500) / 1000.0
        dy = (g("dy  mm") - 500) / 1000.0
        dz = (g("dz  mm") - 500) / 1000.0
        dr = np.deg2rad((g("droll  0.1deg") - 150) / 10.0)
        dp = np.deg2rad((g("dpitch 0.1deg") - 150) / 10.0)
        dyw = np.deg2rad((g("dyaw   0.1deg") - 150) / 10.0)
        T = tft.euler_matrix(dr, dp, dyw, axes="sxyz")
        T[:3, 3] = [dx, dy, dz]
        return T, (dx, dy, dz, np.rad2deg(dr), np.rad2deg(dp), np.rad2deg(dyw))

    @staticmethod
    def msg_to_mat(ts):
        t, q = ts.transform.translation, ts.transform.rotation
        T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    # ── 메인 루프 ─────────────────────────────────────────────────────
    def spin(self):
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            rate.sleep()
            if self.img is None or self.cloud is None or self.K is None:
                continue

            try:
                ts = self.tf_buf.lookup_transform(
                    self.a.camera_frame, self.cloud.header.frame_id,
                    rospy.Time(0), rospy.Duration(0.5))
            except Exception as e:
                rospy.logwarn_throttle(3, "TF %s <- %s : %s",
                                       self.a.camera_frame, self.cloud.header.frame_id, e)
                continue

            T_base = self.msg_to_mat(ts)          # URDF 실측값
            T_delta, dv = self.read_delta()
            T = T_delta.dot(T_base)               # 보정된 optical <- lidar

            pts = np.array(list(point_cloud2.read_points(
                self.cloud, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float32)
            if pts.size == 0:
                continue
            if len(pts) > 60000:                  # 표시용 서브샘플
                pts = pts[np.random.choice(len(pts), 60000, replace=False)]

            # LiDAR -> 카메라 광학좌표
            P = (T[:3, :3].dot(pts.T) + T[:3, 3:4]).T
            fwd = P[:, 2] > 0.15                  # ★ 카메라 뒤쪽 제거
            P = P[fwd]
            if len(P) == 0:
                continue

            rng = np.linalg.norm(P, axis=1)
            rmax = max(1, cv2.getTrackbarPos("max range m", WIN))
            keep = rng < rmax
            P, rng = P[keep], rng[keep]
            if len(P) == 0:
                continue

            uv, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), self.K, self.D)
            uv = uv.reshape(-1, 2)

            vis = self.img.copy()
            h, w = vis.shape[:2]
            u, v = uv[:, 0], uv[:, 1]
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u, v, rng = u[ok].astype(int), v[ok].astype(int), rng[ok]

            colors = cv2.applyColorMap(
                (255 * (1.0 - rng / rmax)).astype(np.uint8).reshape(-1, 1),
                cv2.COLORMAP_JET).reshape(-1, 3)
            for i in range(len(u)):
                cv2.circle(vis, (u[i], v[i]), self.pt_size,
                           tuple(int(c) for c in colors[i]), -1)

            txt = ("dx%+.3f dy%+.3f dz%+.3f m | droll%+.1f dpitch%+.1f dyaw%+.1f deg"
                   % dv)
            cv2.rectangle(vis, (0, 0), (w, 56), (0, 0, 0), -1)
            cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, "points on image: %d / %d" % (len(u), len(pts)),
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(WIN, vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('s'):
                self.save(T, dv)
            elif k == ord('r'):
                for n, m in (("dx  mm", 500), ("dy  mm", 500), ("dz  mm", 500),
                             ("droll  0.1deg", 150), ("dpitch 0.1deg", 150),
                             ("dyaw   0.1deg", 150)):
                    cv2.setTrackbarPos(n, WIN, m)
            elif k == ord(']'):
                self.pt_size = min(6, self.pt_size + 1)
            elif k == ord('['):
                self.pt_size = max(1, self.pt_size - 1)

        cv2.destroyAllWindows()

    # ── 저장: xacro property 형태로 역산 출력 ─────────────────────────
    def save(self, T_opt_lidar, dv):
        pre = self.a.urdf_prefix
        try:
            ts = self.tf_buf.lookup_transform("base_link", self.cloud.header.frame_id,
                                              rospy.Time(0), rospy.Duration(0.5))
        except Exception as e:
            rospy.logerr("base_link <- lidar TF 실패, xacro 역산 불가: %s", e)
            return
        T_base_lidar = self.msg_to_mat(ts)

        # T_base_camlink = T_base_lidar · inv(T_opt_lidar) · T_link_opt
        T_base_camlink = T_base_lidar.dot(np.linalg.inv(T_opt_lidar)).dot(T_LINK_OPT)
        x, y, z = T_base_camlink[:3, 3]
        r, p, yw = np.rad2deg(tft.euler_from_matrix(T_base_camlink, axes="sxyz"))

        block = (
            '\n'
            '  <!-- calib_refine 보정 결과 — 아래 6줄을 URDF에 덮어쓰세요 -->\n'
            '  <xacro:property name="%s_x"     value="%.4f"/>\n'
            '  <xacro:property name="%s_y"     value="%.4f"/>\n'
            '  <xacro:property name="%s_z"     value="%.4f"/>\n'
            '  <xacro:property name="%s_roll"  value="%.2f"/>\n'
            '  <xacro:property name="%s_pitch" value="%.2f"/>\n'
            '  <xacro:property name="%s_yaw"   value="%.2f"/>\n'
            % (pre, x, pre, y, pre, z, pre, r, pre, p, pre, yw)
        )
        print(block)

        out = self.a.out or ("extrinsic_%s.yaml" % pre)
        with open(out, "w") as f:
            f.write("# calib_refine.py 결과\n")
            f.write("# %s_optical_frame <- %s (4x4, row-major)\n"
                    % (pre, self.cloud.header.frame_id))
            f.write("target_frame: %s\n" % self.a.camera_frame)
            f.write("source_frame: %s\n" % self.cloud.header.frame_id)
            f.write("extrinsic_matrix:\n  rows: 4\n  cols: 4\n  data: [\n")
            for row in T_opt_lidar:
                f.write("    %.9f, %.9f, %.9f, %.9f,\n" % tuple(row))
            f.write("  ]\n")
            f.write("base_link_to_%s_link:\n" % pre)
            f.write("  xyz: [%.4f, %.4f, %.4f]\n" % (x, y, z))
            f.write("  rpy_deg: [%.2f, %.2f, %.2f]\n" % (r, p, yw))
        rospy.loginfo("저장: %s", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-frame", required=True, help="예: cam_left_optical_frame")
    ap.add_argument("--image", required=True)
    ap.add_argument("--cloud", default="/ouster/points")
    ap.add_argument("--info", default="")
    ap.add_argument("--urdf-prefix", default="cam", help="xacro property 접두어. 예: left")
    ap.add_argument("--out", default="")
    ap.add_argument("--fx", type=float); ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float); ap.add_argument("--cy", type=float)
    a = ap.parse_args()
    if not a.info and not a.fx:
        sys.exit("--info 또는 --fx/--fy/--cx/--cy 중 하나는 필요합니다")

    rospy.init_node("calib_refine", anonymous=True)
    Refiner(a).spin()


if __name__ == "__main__":
    main()
