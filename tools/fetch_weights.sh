#!/usr/bin/env bash
# 2025 저장소에서 가중치를 가져옵니다.
#
# 이 저장소는 .pt 를 git 에 넣지 않습니다. 같은 파일이 2025 저장소 두 곳에
# 이미 LFS 로 들어가 있고, 여기 또 넣으면 조직 LFS 용량을 세 배로 씁니다.
#
# ★ 파일명 스왑 주의
#     2025 sign.pt        → 내용은 배달표지판  → delivery_sign.pt
#     2025 traffic_sign.pt→ 내용은 콘          → cone.pt
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/weights"
ERP42="${1:-}"      # 2025-CREAM-ERP42 클론 경로
MANDO="${2:-}"      # Mando_Vision 클론 경로

if [[ -z "$ERP42" ]]; then
  echo "사용법: $0 <2025-CREAM-ERP42 경로> [<Mando_Vision 경로>]"
  echo
  echo "예:  $0 ~/git/2025-CREAM-ERP42 ~/git/Mando_Vision"
  echo
  echo "먼저 LFS 를 받아두세요:"
  echo "  cd <저장소> && git lfs install && git lfs pull"
  exit 1
fi

mkdir -p "$DEST"
cp -v "$ERP42/weights/lane.pt"          "$DEST/lane.pt"
cp -v "$ERP42/weights/traffic_sign.pt"  "$DEST/cone.pt"
cp -v "$ERP42/weights/sign.pt"          "$DEST/delivery_sign.pt"
cp -v "$ERP42/weights/traffic_light.pt" "$DEST/traffic_light_20250901.pt"
[[ -n "$MANDO" ]] && cp -v "$MANDO/weights/traffic_light.pt" \
                          "$DEST/traffic_light_20250918.pt"

echo
echo "검증:  python3 tools/check_weights.py weights"
