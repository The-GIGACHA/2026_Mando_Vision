# 2026 Mando Vision

HL FMA(Future Mobility Award) 1/5 — 2026 인지(Perception) 패키지
ROS 패키지명: **`mando_vision_2026`**

> **이 README 는 실제 코드만 설명합니다.**
> 2025 저장소의 README 는 존재하지 않는 노드·launch·가중치를 문서화하고 있었고,
> 그것 때문에 인수인계에 시간이 크게 낭비됐습니다. 여기 적힌 것은 전부 이 저장소에
> 실제로 있는 것입니다. 코드를 지우면 README 도 같이 지워 주세요.

---

## 구성

| 노드 | 파일 | 역할 | 모델 |
|---|---|---|---|
| `lane_detector` | `src/lane_detector.py` | 차선 + 주행가능영역 | YOLOPv2 (TorchScript) |
| `traffic_light_detector` | `src/traffic_light_detector.py` | 신호등 관측 | YOLOv12 |
| `object_detector` | `src/object_detector.py` | 콘 / 배달표지판 | YOLOv12 |
| `detection_monitor` | `src/detection_monitor.py` | 터미널 대시보드 (표시 전용) | — |

## 설계 원칙 — 세 줄

1. **비전은 관측만 발행한다.** GO/STOP 판단은 planning 의 몫입니다.
   신호등 노드는 `label` 을 내보낼 뿐 `Bool` 로 가/불가를 정하지 않습니다.
2. **검출이 없어도 매 주기 발행한다.** 안 보내면 다운스트림이 마지막 값을
   영원히 붙들고 있습니다. 소비자는 `stamp` 로 신선도를 판단합니다.
3. **모르는 것은 STOP.** 클래스 판단은 화이트리스트(무엇이 GO인가)로만 합니다.

## 토픽

### 발행

| 토픽 | 타입 | 주기 |
|---|---|---|
| `/perception/lane` | `std_msgs/String` (JSON) | 15 Hz, 항상 |
| `/perception/traffic_light` | `std_msgs/String` (JSON) | 10 Hz, 항상 |
| `/perception/cone/{left,right}/detections` | `vision_msgs/Detection2DArray` | 10 Hz, 항상 |
| `/perception/delivery_sign/detections` | `vision_msgs/Detection2DArray` | 5 Hz (기본 비활성) |
| `/perception/*/viz/compressed` | `sensor_msgs/CompressedImage` | 5 Hz, **구독자 있을 때만** |

`/perception/traffic_light` 페이로드:

```json
{"detected": true, "label": "green", "conf": 0.93,
 "bbox": [x1, y1, x2, y2], "votes": 4, "window": 5, "stamp": 1234567.89}
```

`bbox` 는 **원본 이미지 좌표**입니다 (ROI 오프셋 복원 완료).
`label` 은 5프레임 중 3표 이상일 때만 채워집니다. 그 외에는 `null`.

### 구독

| 토픽 | 용도 |
|---|---|
| `/cam_front/color/image_raw/compressed` | 차선, 신호등 (D455) |
| `/cam_left/image_raw/compressed` | 콘 좌 (BRIO) |
| `/cam_right/image_raw/compressed` | 콘 우 (BRIO) |
| `/planning/intent` (`String`: `straight`\|`left`) | 모니터 판정 미리보기 |

## 실행

```bash
# 전체
roslaunch mando_vision_2026 perception.launch

# 개별
roslaunch mando_vision_2026 lane.launch
roslaunch mando_vision_2026 traffic_light.launch
roslaunch mando_vision_2026 cone.launch

# TF 만
roslaunch mando_vision_2026 tf.launch
```

## 가중치

**`.pt` 는 git 에 포함되지 않습니다** (`.gitignore`). 같은 파일이 2025 저장소
두 곳에 이미 LFS 로 들어가 있어서, 여기 또 넣으면 조직 LFS 무료 용량(1 GB)을
세 배로 씁니다.

```bash
# 2025 저장소에서 복사 (파일명 스왑 자동 보정)
./tools/fetch_weights.sh ~/git/2025-CREAM-ERP42 ~/git/Mando_Vision

# 검증 — 클래스 매핑이 config/classes.yaml 과 맞는지
python3 tools/check_weights.py weights
```

| 파일 | 내용 | 출처 |
|---|---|---|
| `lane.pt` | YOLOPv2 차선/주행영역 (156 MB) | ERP42 `lane.pt` |
| `cone.pt` | 콘 2종 | ERP42 **`traffic_sign.pt`** ⚠ |
| `delivery_sign.pt` | 배달표지판 7종 | ERP42 **`sign.pt`** ⚠ |
| `traffic_light_20250918.pt` | 신호등 5종 (재학습본, 기본 사용) | Mando_Vision |
| `traffic_light_20250901.pt` | 신호등 5종 (구버전, 비교용) | ERP42 |

> ⚠ **2025 저장소는 `sign.pt` 와 `traffic_sign.pt` 의 이름과 내용이 서로
> 바뀌어 있습니다.** `sign.pt` 안에 배달표지판이, `traffic_sign.pt` 안에 콘이
> 들어 있습니다. 2025 `all_detection.launch` 는 이를 모르고 반대로 물려
> 있었습니다. 여기서는 내용 기준으로 이름을 바로잡았습니다.

## 클래스 매핑

**`config/classes.yaml` 이 단일 진실 소스입니다.** 코드에 클래스를 하드코딩하지
마세요. 모든 노드는 시작 시 실제 모델의 `names` 와 이 파일을 대조하고,
어긋나면 `ROS_FATAL` 후 즉시 종료합니다. 조용히 틀린 판정을 내보내는 것보다
안 뜨는 게 낫습니다.

가중치를 바꿨다면 `classes.yaml` 을 먼저 갱신하세요:

```bash
python3 tools/check_weights.py weights   # 실제 names 추출 + 대조
```

### 신호등 화이트리스트

```yaml
go_straight: [green]        # 직진 허용
go_left:     [green_left]   # 좌회전 허용
```

`left` 는 의미가 확정되지 않아 **의도적으로 제외**되어 있습니다.
2025-09-01 모델의 index 2 는 `red_left` 였고 09-18 재학습본에서 `left` 로
바뀌었습니다. 나머지 4개 클래스가 그대로이므로 단순 리네이밍(= 실제로는
적색+좌회전 = 정지)일 가능성이 높습니다. 확정되면 `classes.yaml` 에 추가하세요.

확인 방법: 좌회전 화살표만 켜진 사진과 적색만 켜진 사진을 넣고 어느 라벨이
나오는지 비교.

## 센서 TF

`urdf/gigacha_sensors.urdf.xacro` 상단의 측정 워크시트를 채우세요.
절차는 `docs/MEASURE.md`.

**`_link` 와 `_optical_frame` 은 다른 프레임입니다.**
카메라 내부행렬 `K` 는 광학 규약(z = 렌즈 방향)을 전제로 합니다.
LiDAR 점을 이미지에 투영할 때는 반드시 `*_optical_frame` 좌표로 변환한 뒤
`K` 를 곱하세요. 2025 융합 코드는 이 구분을 하지 않아 결과가 무의미했습니다.

외부 파라미터 미세보정: `tools/calib_refine.py` (체커보드 없이 1~2 px 수준)

## 요구사항

- Ubuntu 20.04 / ROS Noetic
- CUDA 지원 PyTorch — **YOLOPv2 를 CPU 로 돌리면 1~2 FPS 입니다**
- `ultralytics` — YOLOv12 를 지원하는 버전 (8.3.78 이상). 2025 Dockerfile 의
  `8.3.0` 은 YOLOv12 공개 이전 버전이라 로드되지 않습니다.

```bash
pip install torch torchvision ultralytics opencv-contrib-python
```

`opencv-python` 과 `opencv-contrib-python` 을 동시에 설치하지 마세요.
둘 다 `cv2` 를 제공해서 충돌합니다.

## 이력

`docs/MIGRATION.md` — 어느 파일을 어디서 가져왔고 무엇을 고쳤는지.
