#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  cam_doctor.sh — 카메라가 안 뜰 때 원인을 순서대로 좁힌다 (2대 구성)
#
#  cam_setup.sh 와 역할이 다릅니다.
#    cam_setup.sh : 4대 시절 열거·udev 초안·대역폭 테스트
#                   ★ --udev 는 KERNELS 기반이라 지금 규칙(serial)과 충돌합니다.
#                     쓰지 마십시오.
#    cam_doctor.sh: 대회 구성(cam_front=D455, cam_stopline=C920)이
#                   안 뜰 때의 원인 추적. 아무것도 고치지 않고 진단만 합니다.
#
#  사용:
#      ./tools/cam_doctor.sh          하드웨어 ~ 장치 레벨
#      ./tools/cam_doctor.sh --ros    위 + ROS 토픽까지 (노드 띄운 상태에서)
#
#  원칙: 위에서부터 순서대로 봅니다. 앞 단계가 실패하면 뒷 단계 결과는
#        의미가 없습니다. 첫 [FAIL] 에서 멈추고 그것부터 고치십시오.
# ════════════════════════════════════════════════════════════════════════
set -uo pipefail

C920_SERIAL="FBA4B41F"
D455_SERIAL="238623060437"
RULES="/etc/udev/rules.d/99-gigacha-cameras.rules"
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FAILED=0
ok()   { echo "  [OK]   $*"; }
warn() { echo "  [WARN] $*"; }
bad()  { echo "  [FAIL] $*"; FAILED=$((FAILED+1)); }
fix()  { echo "         └ $*"; }
hdr()  { echo; echo "═══ $* ═══"; }

# ── 0. 진단 도구 ───────────────────────────────────────────────────────
hdr "0. 진단 도구"

MISSING=""
for t in lsusb v4l2-ctl udevadm fuser dmesg; do
  command -v "$t" >/dev/null || MISSING="$MISSING $t"
done
if [ -n "$MISSING" ]; then
  # 도구가 없으면 아래 판정이 거짓 실패로 나옵니다. 먼저 설치하십시오.
  warn "없는 도구:$MISSING"
  fix "sudo apt install usbutils v4l-utils psmisc"
  echo "         ★ 위 도구 없이 나온 아래 [FAIL] 은 신뢰하지 마십시오."
else
  ok "필요 도구 모두 존재"
fi

# ── 1. USB 물리 레벨 ───────────────────────────────────────────────────
hdr "1. USB 연결"

if ! command -v lsusb >/dev/null; then
  warn "lsusb 없음 — 1단계 건너뜀"
else
  if lsusb | grep -qi "046d:08e5\|046d:.*C920\|Logitech.*Webcam"; then
    ok "C920 USB 인식됨"
  else
    bad "C920 이 USB 에 안 보임"
    fix "케이블/포트 교체. 허브 경유면 직결로."
  fi

  if lsusb | grep -qi "8086:0b5c\|Intel.*RealSense\|Intel.*Depth"; then
    ok "D455 USB 인식됨"
  else
    bad "D455 가 USB 에 안 보임"
    fix "USB3 포트에 직결. 허브 경유 금지. 케이블이 USB3(파란 단자)인지 확인."
  fi

  # D455 는 USB2 로 잡혀도 '동작은' 합니다. 그래서 더 위험합니다 —
  # 에러 없이 해상도와 FPS 만 조용히 떨어집니다.
  SPEED=$(lsusb -t 2>/dev/null | grep -A2 -i "uvcvideo\|Intel" \
          | grep -o "5000M\|480M\|10000M" | head -1)
  case "$SPEED" in
    5000M|10000M) ok "USB 링크 속도 $SPEED" ;;
    480M) bad "USB2(480M) 로 잡힘 — 1280x720@30 이 안 나옵니다"
          fix "USB3 포트 + USB3 케이블. BRIO 때 겪은 것과 같은 증상입니다." ;;
    *)    warn "링크 속도 판별 실패 — 'lsusb -t' 를 직접 확인하십시오" ;;
  esac
fi

# ── 2. udev 심볼릭 링크 ────────────────────────────────────────────────
hdr "2. /dev/cam_stopline (C920)"

if [ -e /dev/cam_stopline ]; then
  TARGET=$(readlink -f /dev/cam_stopline)
  ok "심볼릭 링크 존재 → $TARGET"

  # index=0 이 아닌 노드에 링크가 걸리면 usb_cam 이 'cannot open' 으로 죽습니다.
  IDX=$(udevadm info -q property -n /dev/cam_stopline 2>/dev/null | sed -n 's/^ID_V4L_.*INDEX=//p')
  if v4l2-ctl -d /dev/cam_stopline --list-formats 2>/dev/null | grep -q "MJPG"; then
    ok "MJPG 포맷 지원 (실제 영상 노드)"
  else
    bad "MJPG 이 없음 — 메타데이터 노드에 링크가 걸렸을 수 있습니다"
    fix "udev 규칙의 ATTR{index}==\"0\" 확인"
  fi

  SER=$(udevadm info -a -n /dev/cam_stopline 2>/dev/null | sed -n 's/.*ATTRS{serial}=="\(.*\)".*/\1/p' | head -1)
  if [ "$SER" = "$C920_SERIAL" ]; then
    ok "시리얼 일치 ($SER)"
  else
    warn "시리얼이 규칙과 다름: 실제=$SER 기대=$C920_SERIAL"
    fix "카메라를 교체했다면 udev/99-gigacha-cameras.rules 의 serial 을 갱신"
  fi
else
  bad "/dev/cam_stopline 없음 — usb_cam 이 respawn 무한루프에 빠집니다"
  fix "에러가 아니라 '로그 도배'로 나타나므로 정상처럼 보입니다."
  echo
  if [ -f "$RULES" ]; then
    fix "규칙은 설치돼 있음. 재적용:"
    echo "           sudo udevadm control --reload"
    echo "           sudo udevadm trigger --subsystem-match=video4linux"
  else
    fix "규칙이 /etc 에 없음. 설치:"
    echo "           sudo install -m 644 $PKG/udev/99-gigacha-cameras.rules $RULES"
    echo "           sudo udevadm control --reload && sudo udevadm trigger"
  fi
  echo "         └ 그래도 안 생기면 USB 재연결"
fi

# 같은 SYMLINK 를 만드는 규칙이 둘이면 어느 쪽이 이겼는지 알 수 없게 됩니다.
CONFLICT=$(grep -rl "cam_stopline\|cam_left\|cam_right" /etc/udev/rules.d/ 2>/dev/null | grep -v "99-gigacha-cameras.rules")
if [ -n "$CONFLICT" ]; then
  bad "충돌 가능 규칙 발견:"
  echo "$CONFLICT" | sed 's/^/           /'
  fix "같은 SYMLINK 를 놓고 경쟁합니다. 하나만 남기십시오."
else
  ok "충돌하는 udev 규칙 없음"
fi

# ── 3. D455 ────────────────────────────────────────────────────────────
hdr "3. D455"

# realsense2_camera 는 /dev/video* 가 아니라 USB 를 직접 엽니다.
# 따라서 udev SYMLINK 와 무관하고, rs-enumerate-devices 가 유일한 확인 수단입니다.
if command -v rs-enumerate-devices >/dev/null; then
  RS=$(timeout 15 rs-enumerate-devices -s 2>&1)
  if echo "$RS" | grep -q "$D455_SERIAL"; then
    ok "D455 인식 (serial $D455_SERIAL)"
  elif echo "$RS" | grep -qi "D455\|D4"; then
    warn "D455 는 보이지만 시리얼이 다름"
    echo "$RS" | sed 's/^/           /'
    fix "카메라를 교체했다면 udev 규칙 주석의 serial 을 갱신"
  else
    bad "rs-enumerate-devices 가 장치를 못 찾음"
    echo "$RS" | head -5 | sed 's/^/           /'
    fix "USB 재연결 후 재시도. 그래도 안 되면 전원 사이클."
  fi
else
  warn "rs-enumerate-devices 없음 (librealsense2-utils 미설치)"
fi

# ── 4. 장치 점유 ───────────────────────────────────────────────────────
hdr "4. 장치를 잡고 있는 프로세스"

BUSY=0
for d in /dev/cam_stopline /dev/video*; do
  [ -e "$d" ] || continue
  P=$(fuser "$d" 2>/dev/null)
  if [ -n "$P" ]; then
    BUSY=1
    warn "$d 사용 중: PID$P"
    ps -o pid=,comm=,args= -p $P 2>/dev/null | cut -c1-100 | sed 's/^/           /'
  fi
done
if pgrep -f "realsense2_camera\|rs-" >/dev/null; then
  BUSY=1
  warn "realsense 관련 프로세스가 이미 실행 중"
  pgrep -af "realsense2_camera|rs-" | cut -c1-100 | sed 's/^/           /'
fi
[ "$BUSY" = 0 ] && ok "점유 중인 프로세스 없음"
[ "$BUSY" = 1 ] && fix "이전 roslaunch 가 안 죽었을 수 있습니다: pkill -f roslaunch"

# ── 5. 커널 로그 ───────────────────────────────────────────────────────
hdr "5. 최근 USB 커널 메시지"

DMESG=$(dmesg 2>/dev/null | tail -400 | grep -i -e "usb.*disconnect" -e "usb.*reset" \
        -e "uvcvideo" -e "no space left" -e "cannot submit" -e "device descriptor read" | tail -12)
if [ -z "$DMESG" ]; then
  ok "특이 메시지 없음"
else
  echo "$DMESG" | sed 's/^/         /'
  echo
  echo "         해석:"
  echo "           'No space left on device'  → USB 대역폭 부족. 해상도↓ 또는 다른 Bus 로"
  echo "           'device descriptor read'   → 케이블/전원 불량"
  echo "           반복되는 disconnect/reset  → 접촉 불량. 주행 진동 중이면 특히 의심"
fi

# ── 6. ROS ─────────────────────────────────────────────────────────────
if [ "${1:-}" = "--ros" ]; then
  hdr "6. ROS 토픽 (노드를 띄운 상태여야 합니다)"

  if ! timeout 3 rostopic list >/dev/null 2>&1; then
    bad "roscore 에 연결 불가"
  else
    for t in /cam_front/color/image_raw/compressed /cam_stopline/image_raw/compressed; do
      if timeout 3 rostopic list 2>/dev/null | grep -qx "$t"; then
        HZ=$(timeout 12 rostopic hz "$t" 2>/dev/null | sed -n 's/.*average rate: \([0-9.]*\).*/\1/p' | tail -1)
        if [ -n "$HZ" ]; then
          ok "$t  ${HZ} Hz"
        else
          bad "$t 는 있는데 발행이 없음"
          fix "노드는 떴지만 프레임이 안 옵니다. 위 1~4 단계를 다시 보십시오."
        fi
      else
        bad "$t 없음"
        # /compressed 는 image_transport 플러그인이 있어야 생깁니다.
        RAW="${t%/compressed}"
        if timeout 3 rostopic list 2>/dev/null | grep -qx "$RAW"; then
          fix "$RAW 는 있는데 /compressed 가 없습니다."
          fix "sudo apt install ros-noetic-compressed-image-transport"
          fix "★ 인지 노드는 /compressed 만 구독합니다. 이게 없으면 전부 멉니다."
        fi
      fi
    done
  fi
fi

# ── 요약 ───────────────────────────────────────────────────────────────
echo
if [ "$FAILED" = 0 ]; then
  echo "═══ 실패 항목 없음 ═══"
else
  echo "═══ [FAIL] $FAILED 건 — 가장 위의 것부터 처리하십시오 ═══"
fi

cat <<'EOF'

── roslaunch 에러 문구 대조표 ────────────────────────────────────────
  "Device or resource busy"          이미 다른 프로세스가 잡음 → 4단계
  "failed to set power state"        USB 재연결 필요. 전원 사이클도 시도
  "Resource temporarily unavailable" initial_reset 실패 → 물리 재연결
  "No device connected"              D455 미인식 → 3단계
  "cannot open /dev/cam_stopline"    심볼릭 링크가 메타데이터 노드 → 2단계
  cam_stopline 노드가 계속 재시작     /dev/cam_stopline 없음 → 2단계
  "VIDIOC_STREAMON: No space left"   USB 대역폭 부족 → 5단계
  영상은 나오는데 7 FPS               C920 자동노출. 재연결하면 v4l2 설정이
                                     초기화됩니다. 노드를 다시 띄우면 launch 의
                                     autoexposure:=false 가 재적용됩니다.
EOF
exit $FAILED
