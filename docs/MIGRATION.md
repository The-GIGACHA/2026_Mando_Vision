# 2025 → 2026 이관 기록

## 출처 저장소

| 저장소 | 패키지명 | 커밋 | 성격 |
|---|---|---|---|
| `The-GIGACHA/2025-CREAM-ERP42` | `cream_erp` | 9개, 2025-09-01 | 개발 저장소 (전체 인지) |
| `The-GIGACHA/Mando_Vision` | `cream_hl` | 1개(squash), 2025-09-18 | 대회 직전 스냅샷 (축소판) |

**Mando_Vision 이 17일 뒤 버전이고, 차선·콘·표지판 노드를 전부 들어냈습니다.**
대신 정지선 검출 + 모니터 + Docker 를 추가했습니다.
→ 작년 팀이 대회 직전에 차선·콘·표지판 검출을 포기했다는 뜻입니다.
   왜 버렸는지(정확도? FPS?) 확인하면 올해 같은 벽을 피할 수 있습니다.

## 파일별 이관

| 2026 파일 | 출처 | 처리 |
|---|---|---|
| `src/lane_detector.py` | ERP42 `src/lane_detector.py` | 수정 이관 |
| `src/traffic_light_detector.py` | 두 저장소 모두 참고 | **재작성** |
| `src/object_detector.py` | ERP42 `cone_detector.py` + `traffic_sign_detector.py` | 통합 + 수정 |
| `src/detection_monitor.py` | Mando_Vision | 수정 이관 |
| `utils/utils.py` | 양쪽 동일 | **그대로** (YOLOPv2 후처리) |
| `utils/class_map.py` | 신규 | 클래스 매핑 검증 |
| `utils/infer_loop.py` | 신규 | 공통 안전 장치 |
| `config/classes.yaml` | 신규 | 단일 진실 소스 |
| `urdf/`, `tools/calib_refine.py`, `docs/MEASURE.md` | 신규 | 센서 TF·캘리브레이션 |
| 정지선 검출 3종 | Mando_Vision | **폐기** — 아래 참조 |
| `threshold_tuner*.py` | Mando_Vision | **폐기** — 튜닝 대상이 사라짐 |
| `Dockerfile`, `docker/` | Mando_Vision | 보류 (ultralytics 버전 갱신 후 이관) |

## 고친 것 — 심각도 순

### 🔴 1. 클래스 매핑 불일치

```
sign.pt        → 실제 내용: 배달표지판 7종 (0_delivery_a1 … 8_parkinglot)
traffic_sign.pt→ 실제 내용: 콘 2종        (6_cone_blue, 7_cone_yellow)
```

**파일명과 내용이 서로 바뀌어 있었고, 2025 `all_detection.launch` 는 그대로
반대로 물려 있었습니다.** `cone_detector` 가 배달표지판 모델을,
`traffic_sign_detector` 가 콘 모델을 로드하고 있었습니다.

→ `weights/cone.pt`, `weights/delivery_sign.pt` 로 내용 기준 개명.
→ `config/classes.yaml` 을 단일 진실 소스로 두고, 노드가 시작 시 실제
   `model.names` 와 대조해 어긋나면 `ROS_FATAL` 로 종료.

이름 앞의 숫자는 원본 통합 데이터셋(0~8) 인덱스이고 **모델 내부의 실제
인덱스와 다릅니다.** 예: `delivery_sign.pt` 의 로컬 6번 = `8_parkinglot`.
코드는 반드시 로컬 인덱스를 씁니다.

### 🔴 2. 신호등 판정 — 부분 문자열 매칭

`detection_monitor.py` L101:

```python
if "green" in traffic_color.lower():   # "green" in "green_left" → True
    return "GOGOGO", "GREEN"
```

**좌회전 신호에서 직진 GO 판정.**
→ 화이트리스트 정확 매칭 + `intent`(straight/left) 분리.

### 🔴 3. 신선도 미검사 — 오래된 신호로 계속 GO

`get_action_decision()` 이 `last_update` 를 보지 않았습니다. 화면에는
`TIMEOUT` 이라 표시하면서 판정은 10초 전 초록불을 계속 썼습니다.
→ `FRESH_SEC = 0.5` 초과 시 무조건 STOP.

### 🔴 4. 검출 없으면 미발행 → stale 데이터

`/traffic_light`, `/planning/cone_data`, `lane_points`, `traffic_signs` 등
**5개 토픽 전부** "검출이 있을 때만" 발행했습니다.
→ `HeartbeatPublisher` 로 매 주기 발행. 없으면 `detected: false`.

### 🔴 5. 런타임 pip 설치

`cone_detector.py` L20:

```python
except ImportError:
    os.system("pip install ultralytics")
```

대회장에 네트워크가 없으면 여기서 멈춥니다.
**규정 12번 — 1분 이상 움직임 없으면 DNF.**
→ 삭제. 임포트 실패 시 `ROS_FATAL` 후 종료.

### 🔴 6. ROI 좌표 미복원

`traffic_sign_detector.py` 는 ROI 로 크롭해 추론하고 bbox 를 **ROI 기준
좌표 그대로** 발행했습니다. 융합 노드가 그 bbox 로 LiDAR 점을 매칭하므로,
ROI 를 전체 이미지가 아닌 값으로 두는 순간 3D 위치가 통째로 틀립니다.
2025 launch 가 `[0,1080,0,720]`(전체)이라 우연히 안 드러났습니다.
→ 오프셋을 더해 원본 좌표로 복원.

### 🟠 7. ROI 미구현

`traffic_light_detector.py` 는 launch/config 로 `roi` 를 받으면서
**코드가 읽지도 쓰지도 않았습니다.** ROI 기능이 문서에만 존재했습니다.
→ 실제 구현 + 좌표 복원.

### 🟠 8. 프레임당 여러 번 판정

박스마다 `Bool` 을 발행해 **마지막 박스가 이기는 레이스**였습니다.
빨강+초록화살표가 동시에 잡히면 NMS 정렬 순서가 판정을 결정했습니다.
→ 프레임당 최고 신뢰도 1개만 집계 + 5프레임 중 3표 다수결.

### 🟠 9. 콜백 내 추론 / buff_size 누락

`traffic_light_detector.py` 의 `Subscriber` 에 `queue_size` 조차 없었습니다
(기본 무한 큐). 추론이 밀리면 지연이 무한 누적됩니다.
`queue_size=1` 만으로도 부족합니다 — `buff_size` 를 프레임보다 크게 주지
않으면 rospy 가 소켓 버퍼에 쌓아둡니다.
→ `LatestFrame` (queue_size=1, buff_size=2**24) + 타이머 추론.
   ※ `lane_detector.py` 는 2025 에서도 이미 제대로 되어 있었습니다.

### 🟠 10. device 기본값이 CPU

`lane_detector.py` L30 과 `lane_detection_config.yaml` 이 `device: "cpu"`.
launch 로 `cuda:0` 을 주지만 직접 실행하면 CPU 로 돌고,
YOLOPv2 CPU 는 1~2 FPS 입니다.
→ 기본 `auto`(CUDA 있으면 CUDA) + CPU 로 떨어지면 경고 배너.

### 🟠 11. 시각화 무조건 인코딩

구독자가 없어도 매 프레임 JPEG 인코딩(1280×720 기준 5~8 ms).
`lane_detector` 는 raw Image 3개(mono8 921 KB ×2 + bgr8 2.7 MB)를 발행해
30 fps 기준 약 135 MB/s.
→ `get_num_connections() > 0` 체크 + 5 Hz throttle + 압축 발행.

### 🟠 12. 카메라 3대 순차 처리

`cone_detector.py` 가 한 노드·한 스레드에서 cam1/2/3 콜백을 처리했습니다.
→ 카메라 1대 = 노드 1개. launch 에서 인스턴스를 여러 개 띄웁니다.

### 🟡 13. 색을 카메라 위치로 결정

2025 `gigacha_sensor_fusion.cpp` 가 좌측 카메라 = 파랑, 우측 = 노랑으로
하드코딩하고 `ObjectHypothesisWithPose.id` 를 버렸습니다.
검출 노드(`cone_detector.py`)는 id 를 제대로 넣고 있었으니 **융합 쪽 문제**입니다.
→ 이 저장소는 모델이 분류한 id 를 그대로 발행합니다. 융합 코드도 고치세요.

### 🟡 14. 문서와 코드 불일치

2025 README 는 존재하지 않는 노드·launch·가중치를 문서화했습니다.
→ 2026 README 는 실제 코드만 설명합니다.

## 폐기한 것

| 대상 | 이유 |
|---|---|
| `stop_line_detector_front/back/video.py` | ROI 내 흰 픽셀 비율 ≥ 80% 방식. 고정 임계값(front 215 / back 153)이 카메라만 바뀌어도 안 맞는 수준. 역광·그늘·젖은 노면·횡단보도에서 붕괴. **YOLOPv2 lane mask + BEV(IPM) 수평 성분 검출로 대체 예정** |
| `threshold_tuner*.py` | 위 방식이 사라지므로 튜닝 대상 없음 |
| 죽은 함수 4종 | `detect_stop_lines_by_parallel_lines`, `detect_line_stop_lines`, `detect_rectangular_stop_lines_old`, `merge_detections` — 호출되지 않고, `self.hough_threshold` 등 `__init__` 에 없는 속성을 참조. 호출하면 즉시 `AttributeError` 인데 try/except 가 조용히 `[]` 반환 |

## 아직 안 한 것

- [ ] `lane.pt` 도메인 검증 — 실차 도로 영상으로 학습된 모델을 1/5 스케일
      차량(카메라 높이 0.2~0.4 m)에 쓰는 것이라 소실점 위치가 완전히 다릅니다.
      **규정 2번 차로준수는 유일하게 탈락과 직결되는 인지 항목입니다.**
- [ ] `left` 클래스 의미 확정 (`config/classes.yaml` 의 note 참조)
- [ ] 정지선 → YOLOPv2 lane mask + BEV 파이프라인 구현
- [ ] TensorRT 변환 — YOLOPv2 는 3-head TorchScript 라 ONNX export 가 까다롭습니다.
      `split_for_trace_model` 의 anchor decode 를 그래프 밖으로 빼야 합니다. 일찍 시작하세요.
- [ ] `gigacha_lidar` 확보 — 콘 융합의 입력(`/gigacha/lidar/bounding_boxes`)을
      만드는 패키지. 없으면 융합 노드가 아무것도 못 합니다.
- [ ] 2025-GIGACHA-FUSION 의 `projectToCamera` 재작성
      (`Zc > 0` 체크 · 왜곡계수 · 이미지 경계 — 세 가지가 다 빠져 있습니다)
- [ ] `Dockerfile` ultralytics 버전 갱신 (8.3.0 → YOLOv12 지원 버전)
