#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  play_bag.sh — 인지 노드를 bag 으로 돌려볼 때 쓴다.
#
#  ★ 이게 왜 필요한가
#    주행 bag 에는 그날의 /perception/* 결과가 같이 녹화돼 있다.
#    그냥 `rosbag play bag.bag` 하면 녹화분과 지금 노드 출력이 같은 토픽에
#    둘 다 실려서, 발행율이 2배로 보이고 rqt/monitor 가 옛 결과를 섞어 보여준다.
#    2026-09-20 에 실제로 이걸로 한참 헤맸다. 15 Hz 노드가 30 Hz 로 보였다.
#      /perception/sign_car  Publishers: /obstacle_detector, /play_1789...
#    그래서 센서와 게이트 입력만 골라 재생한다.
#
#  사용
#      tools/play_bag.sh <bag> [rosbag play 추가인자...]
#      tools/play_bag.sh ~/bags/mando/h_20260919_150248.bag -s 205 -u 195
#      tools/play_bag.sh ~/bags/mando/traffic_light_20260919_135705.bag
#
#  같이 띄울 것 (별도 터미널, ★ roslaunch 전에 use_sim_time)
#      rosparam set use_sim_time true
#      roslaunch mando_vision_2026 perception.launch stopline:=false \
#          depth_topic:=/cam_front/aligned_depth_to_color/image_raw/compressedDepth
#      rqt_image_view /perception/obstacle/viz/compressed
#
#  ★ 녹화된 결과를 '비교용' 으로 같이 보고 싶으면 --with-recorded 를 준다.
#    그때는 /perception/* 이 /bag/perception/* 으로 들어온다 (충돌 없음).
# ════════════════════════════════════════════════════════════════════════
set -euo pipefail

[[ $# -ge 1 ]] || { sed -n '2,30p' "$0"; exit 1; }
BAG=$1; shift

WITH_REC=0
ARGS=()
for a in "$@"; do
  if [[ "$a" == "--with-recorded" ]]; then WITH_REC=1; else ARGS+=("$a"); fi
done

# 노드가 실제로 구독하는 것만. 늘리려면 여기에 추가한다.
TOPICS=(
  /cam_front/color/image_raw/compressed
  /cam_front/color/camera_info
  /cam_front/aligned_depth_to_color/image_raw/compressedDepth
  /cam_front/aligned_depth_to_color/camera_info
  /cam_stopline/image_raw/compressed
  /global_active_map          # 미션 게이트
  /global_nearest_idx         # 미션 게이트
  /tf_static
)

REMAP=()
if [[ $WITH_REC -eq 1 ]]; then
  for t in $(rosbag info -y -k topics "$BAG" | sed -n 's/^- topic: //p' | grep '^/perception/'); do
    TOPICS+=("$t"); REMAP+=("$t:=/bag$t")
  done
fi

HAVE=()
AVAIL=$(rosbag info -y -k topics "$BAG" | sed -n 's/^- topic: //p')
for t in "${TOPICS[@]}"; do grep -qx -- "$t" <<<"$AVAIL" && HAVE+=("$t"); done

echo "재생 토픽 ${#HAVE[@]}개:"; printf '   %s\n' "${HAVE[@]}"
[[ ${#REMAP[@]} -gt 0 ]] && { echo "녹화분 리맵:"; printf '   %s\n' "${REMAP[@]}"; }
echo
exec rosbag play --clock "${ARGS[@]+"${ARGS[@]}"}" "$BAG" \
     --topics "${HAVE[@]}" ${REMAP[@]+"${REMAP[@]}"}
