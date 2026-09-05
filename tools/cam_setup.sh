#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  cam_setup.sh — USB 카메라 4대 열거 / 진단 / udev 규칙 생성 / AF·AE 잠금
#
#  BRIO 2대는 VID:PID 가 같아서 by-id 로 구분이 안 됩니다.
#  물리 포트(KERNELS) 기준 udev 규칙이 유일한 해법입니다.
#
#  사용:
#    ./cam_setup.sh                 진단만 (아무것도 안 고침)
#    ./cam_setup.sh --udev          udev 규칙 초안을 stdout 으로 출력
#    ./cam_setup.sh --lock /dev/videoN   AF/AE/WB 잠금
#    ./cam_setup.sh --stream        4대 동시 스트림 대역폭 테스트
# ════════════════════════════════════════════════════════════════════════
set -uo pipefail
MODE="${1:-diag}"

need() { command -v "$1" >/dev/null || { echo "필요: $1  (sudo apt install $2)"; exit 1; }; }

# ── 1. 진단 ────────────────────────────────────────────────────────────
diag() {
  need v4l2-ctl v4l-utils
  echo "════════ USB 컨트롤러 트리 ════════"
  echo "★ 카메라들이 서로 다른 Bus 에 흩어져 있어야 대역폭 경합이 없습니다."
  echo "  같은 Bus 에 BRIO 2대가 몰려 있으면 한 대를 다른 포트로 옮기세요."
  lsusb -t
  echo
  echo "════════ 비디오 장치 ════════"
  v4l2-ctl --list-devices 2>/dev/null
  echo
  for d in /dev/video*; do
    [ -e "$d" ] || continue
    # index=0 인 노드만 실제 캡처 장치입니다 (나머지는 UVC 메타데이터 노드)
    idx=$(udevadm info -q property -n "$d" 2>/dev/null | sed -n 's/^ID_V4L_PRODUCT=//p')
    cap=$(v4l2-ctl -d "$d" --all 2>/dev/null | sed -n 's/^\s*Video Capture$/YES/p' | head -1)
    fmts=$(v4l2-ctl -d "$d" --list-formats 2>/dev/null | sed -n "s/.*'\(.*\)'.*/\1/p" | tr '\n' ' ')
    [ -z "$fmts" ] && continue
    kern=$(udevadm info -a -n "$d" 2>/dev/null | sed -n 's/.*KERNELS=="\([0-9]*-[0-9.]*\)".*/\1/p' | head -1)
    echo "── $d"
    echo "     제품     : ${idx:-?}"
    echo "     포트     : ${kern:-?}      ← udev 규칙의 KERNELS 값"
    echo "     포맷     : ${fmts:-none}"
    printf "     현재     : "
    v4l2-ctl -d "$d" --get-fmt-video 2>/dev/null | sed -n 's/.*Width\/Height *: *\(.*\)/\1/p' | tr -d '\n'
    v4l2-ctl -d "$d" --get-parm 2>/dev/null | sed -n 's/.*(\(.*\) fps).*/  \1 fps/p'
    echo
  done
  echo "════════ 체크리스트 ════════"
  echo "  [ ] MJPG 가 목록에 있는가? 없으면 무압축(YUYV)뿐이라 1대만 꽂아도"
  echo "      USB3 대역폭의 1/3 을 씁니다."
  echo "  [ ] 4대가 서로 다른 Bus 에 있는가?"
  echo "  [ ] D455 는 USB3 포트에 직결되어 있는가? (허브 경유 금지)"
}

# ── 2. udev 규칙 초안 ──────────────────────────────────────────────────
udev() {
  echo "# /etc/udev/rules.d/99-gigacha-cameras.rules"
  echo "#"
  echo "# ★ KERNELS = 물리 포트 위치입니다. 포트를 바꿔 꽂으면 규칙도 바꿔야 합니다."
  echo "#   카메라마다 꽂는 포트를 고정하고 케이블에 라벨을 붙이세요."
  echo "# ★ ATTR{index}==\"0\" 를 빼면 UVC 메타데이터 노드에도 링크가 걸려"
  echo "#   열리지 않는 장치가 생깁니다. 반드시 넣으세요."
  echo
  for d in /dev/video*; do
    [ -e "$d" ] || continue
    fmts=$(v4l2-ctl -d "$d" --list-formats 2>/dev/null | sed -n "s/.*'\(.*\)'.*/\1/p")
    [ -z "$fmts" ] && continue
    kern=$(udevadm info -a -n "$d" 2>/dev/null | sed -n 's/.*KERNELS=="\([0-9]*-[0-9.]*\)".*/\1/p' | head -1)
    name=$(udevadm info -q property -n "$d" 2>/dev/null | sed -n 's/^ID_V4L_PRODUCT=//p')
    echo "# $d  ($name)"
    echo "SUBSYSTEM==\"video4linux\", KERNELS==\"${kern}\", ATTR{index}==\"0\", SYMLINK+=\"cam_CHANGE_ME\""
  done
  echo
  echo "# 권장 심볼릭 이름: cam_front(D455) / cam_stopline(C920) / cam_left / cam_right"
  echo "#"
  echo "# 적용:"
  echo "#   sudo cp 99-gigacha-cameras.rules /etc/udev/rules.d/"
  echo "#   sudo udevadm control --reload-rules && sudo udevadm trigger"
  echo "#   ls -l /dev/cam_*"
}

# ── 3. AF/AE/WB 잠금 ───────────────────────────────────────────────────
lock() {
  local d="${2:-}"
  [ -z "$d" ] && { echo "사용: $0 --lock /dev/videoN"; exit 1; }
  echo "── $d 컨트롤 목록"
  v4l2-ctl -d "$d" --list-ctrls
  echo
  echo "── 잠금 시도 (커널 버전에 따라 이름이 다릅니다. 실패는 무시하세요)"
  for c in "focus_automatic_continuous=0" "focus_auto=0" \
           "focus_absolute=0" \
           "auto_exposure=1" "exposure_auto=1" \
           "white_balance_automatic=0" "white_balance_temperature_auto=0"; do
    v4l2-ctl -d "$d" -c "$c" 2>/dev/null && echo "   OK   $c"
  done
  echo
  echo "★ 캘리브레이션은 '운용할 초점값 그대로' 해야 합니다."
  echo "  캘리 때만 근거리에 맞췄다가 주행 때 무한대로 돌리면 fx 가 달라져"
  echo "  캘리 결과가 무효가 됩니다."
}

# ── 4. 동시 스트림 테스트 ──────────────────────────────────────────────
stream() {
  need ffmpeg ffmpeg
  local devs=(); for d in /dev/video*; do
    v4l2-ctl -d "$d" --list-formats 2>/dev/null | grep -q MJPG && devs+=("$d")
  done
  echo "MJPG 지원 장치: ${devs[*]}"
  echo "4대를 동시에 5초간 열어봅니다. 'No space left on device' 가 뜨면"
  echo "USB 대역폭 부족입니다 — 해상도를 낮추거나 다른 Bus 로 옮기세요."
  echo
  local pids=()
  for d in "${devs[@]}"; do
    ffmpeg -hide_banner -loglevel error -f v4l2 -input_format mjpeg \
           -video_size 1280x720 -framerate 30 -i "$d" -t 5 -f null - &
    pids+=($!); echo "  열기: $d (pid $!)"
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  echo
  [ "$fail" = 0 ] && echo "✓ 4대 동시 스트림 성공" \
                  || echo "✗ 실패 — 위 에러 메시지를 확인하세요"
}

case "$MODE" in
  --udev)   udev ;;
  --lock)   lock "$@" ;;
  --stream) stream ;;
  *)        diag ;;
esac
