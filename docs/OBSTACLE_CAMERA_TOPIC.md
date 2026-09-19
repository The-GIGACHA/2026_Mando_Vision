# 카메라 장애물 좌표 — 제어로 넘기는 규약

2026-09-15 결정 (방식 B): 카메라 쪽이 장애물을 **차량 좌표(base_link)** 로 바꿔서 보낸다.
2026-09-17 변경:
- 제어(`fma_15_control/controller/control/path_collision_checker.py`)는 **bbox 를 다시 투영하지 않는다.**
  카메라 기하(K·높이·pitch)가 인지·제어 두 벌이면 한쪽만 고쳤을 때 에러 없이 좌표가 어긋나기 때문이다.
- 뎁스 점(`/obstacles/camera`)은 **S자 코스에서만** 낸다.

| 구간 | 제어가 쓰는 카메라 입력 | 발행 노드 |
|---|---|---|
| `S_COURSE` | `/obstacles/camera` 뎁스 점 (안 오면 아래 x/y 로 대신 + 경고) | `obstacle_depth` |
| 그 외 | `/perception/obstacle` items 의 `x`, `y`, `width` | `ground_markers` |

## `/perception/obstacle` (JSON, 전 구간)

`ground_markers` 가 bbox 밑변을 평지 가정으로 투영한 값 (`utils/obstacle_ground.py`).
- `x` = 밑변(가까운 면)의 전방거리, `y` = 좌우 중심, `width` = 좌우 폭. base_link, m.
- 제어는 `x` 에서 `y ± width/2` 를 5점으로 펼쳐 판정한다. `x` 가 null 인 항목은 버린다.
- ★ 밑동이 화면 아래로 잘린 장애물(base_link ≈2.70 m 안)은 `x` 가 2.70 m 로 뭉개진다.
  높이 0.40 m T870 은 base_link ≈1.91 m 보다 가까우면 화면에서 통째로 사라진다.

## `/obstacles/camera` (PointCloud2, S자 코스만)

| 항목 | 값 |
|---|---|
| 타입 | `sensor_msgs/PointCloud2` |
| `header.frame_id` | `base_link` (★ 뒷차축 중심, x 전방 +, y 좌측 +, z 위 +, REP-103) |
| `header.stamp` | **영상 캡처 시각** (컬러 프레임의 stamp). 발행 시각이 아니다 |
| 필드 | `x`, `y`, `z` (float32, 미터) |
| 주기 | 검출 주기마다. **영상을 처리했으면 장애물이 없어도 빈 cloud** |
| 넣는 장애물 | `T870` (콘을 대신 세울 때는 `cone_blue`, `cone_yellow` — `obstacle.launch cone:=true`) |

### 장애물 하나당 점

제어는 "경로 튜브(좌우 0.7 m) 안에 3점 이상"을 막힘으로 보고, 점들의 좌우 끝으로 비킬 폭을 계산한다.
그래서 bbox 를 가로 5칸으로 나눠 칸마다 한 점(가까운 면)을 낸다.

### 뎁스를 그대로 믿지 않는다

D455 뎁스는 역광·반사·저질감 면에서 구멍이 나거나 틀린 거리를 낸다. 박스마다 지면 투영과 대조한다
(`utils/obstacle_depth.py`).

1. 정렬 뎁스(`/cam_front/aligned_depth_to_color/image_raw`) 에서 bbox 아래쪽 절반, 칸마다
   유효 픽셀의 **30 백분위 전방거리**(가까운 면)를 잡는다. 유효 픽셀 20% 미만인 칸은 버린다.
2. 픽셀 + 뎁스 → base_link 는 **`utils/ground.py` 와 같은 카메라 모델**로 옮긴다.
   두 값의 차이가 순수하게 '뎁스 거리 vs 평지 가정 거리' 가 되게 하려는 것이다.
   URDF(TF)와의 일치는 노드 시동 때 한 번 대조해 로그로 낸다.
3. 판정

| used | reason | 조건 | 내는 점 |
|---|---|---|---|
| depth | `ok` | \|뎁스 x − 지면 x\| ≤ max(0.30 m, 0.15 × 지면 x, pitch ±0.5° 거리폭) | 뎁스 칸 점 |
| depth | `cut_ok` | 밑동이 잘렸고 뎁스 x ≤ 지면 x(≈2.70) + 0.30 | 뎁스 칸 점 |
| ground | `mismatch` | 허용오차 밖 (뎁스 오인식 의심) | 지면 투영 밑변 5점 |
| ground | `few_valid` | 유효 칸 3개 미만 (구멍) | 지면 투영 밑변 5점 |
| ground | `far` | 지면 x > 8 m | 지면 투영 밑변 5점 |
| ground | `no_depth` | 영상 시각 ±50 ms 안에 뎁스 프레임 없음 | 지면 투영 밑변 5점 |
| ground | `cut_far` | 밑동이 잘렸는데 뎁스가 더 멀다 | 지면 투영 밑변 5점 (더 가까운 값) |

항목별 판정은 `/perception/obstacle_depth` (JSON) 에 싣고 `bev_viz.py` 가 겹쳐 그린다.

★ S자 코스 좌/우 Bool(`/perception/s_course/right`)은 이 뎁스와 무관하게 지면 투영으로 판정한다.
  한 번 커밋하면 안 뒤집히는 판정에 역광 뎁스 오인식이 들어가면 안 된다.

### pitch 감시

대조는 `~ground/cam_pitch_deg` 가 맞다는 전제 위에 서 있다. 노드가 1 Hz 로 차 앞 노면 뎁스에
평면을 맞춰 카메라 높이·pitch 를 재고, 최근 10회 중앙값이 설정과 1° 넘게 다르면 경고한다.

★ 2026-09-16 송도 bag (`~/bags/mando/songdo_full_20260916_005609.bag`) 실측:
노면 평면 높이 0.974 m, **pitch −2.75°** (당시 URDF −1.0°). 지면 투영이 5 m 에서 약 0.4~1 m 짧게 나와
S코스 콘 43개 중 17개가 `mismatch` 였다. pitch −2.75°·높이 1.000 m 로 다시 돌리면 전 구간 304개 중
`mismatch` 5%, 뎁스−지면 x 차이 중앙값 −0.02 m. → 2026-09-17 URDF·ground.py 를 −2.75° 로 바꿨다.

```bash
python3 tools/bag_obstacle_depth.py BAG                          # 설정 pitch 로
python3 tools/bag_obstacle_depth.py BAG --pitch -3.0 --sheet /tmp/s.jpg      # 다른 pitch 로 비교
```

## 끊김 표시

- 미션 게이트가 꺼져 있거나(S자 코스 밖) 검출 영상이 안 오면 **발행하지 않는다**.
  빈 cloud 는 "봤는데 장애물 없음" 이다. 제어는 1초 동안 안 오면 끊김으로 보고 x/y 로 대신한다.
- GRM 토픽이 없으면(단독 시험·bag 에 GRM 없음) 항상 켜고 경고한다.

## 확인

```bash
rostopic hz /obstacles/camera                         # S자 코스 안에서만 ≈ 검출 주기
rostopic echo -n1 /obstacles/camera/header            # frame_id: base_link, stamp ≈ 영상 시각
rostopic echo /perception/obstacle_depth              # 항목별 used / reason, plane
rosrun mando_vision_2026 bev_viz.py  →  rqt_image_view /perception/bev/compressed
rostopic echo /obstacle_extent                        # [y_min, y_max, 전방거리] — 줄자와 비교
```

장애물을 줄자로 2 m / 3 m / 5 m, 좌우 ±0.5 m 에 두고 `/obstacle_extent` 와 비교한다.
`/obstacle_extent` 의 전방거리는 경로를 따라 **뒷차축부터** 잰 값이다 (앞범퍼는 +1.10 m).
역광에서 같은 시험을 하고 `bag_obstacle_depth.py` 로 `mismatch` / `few_valid` 비율을 보면
뎁스를 얼마나 믿을 수 있는지가 나온다.
