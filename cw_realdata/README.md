# cw_realdata — 리얼 녹음 추론 (2026-09-08)

`clipped_audio/` : 8개 모노 wav (44.1kHz, 1s), 리그 방위각 45°~360° (45° 간격 링).
`infer_realdata.py` : 캠페인 최고 8ch 모델 `0820_sslam_llrd_r8vd_rep` (sslam+LLRD+vdrop,
Replica test 0.2363)로 추론. 스크립트 docstring에 전 가정 명시.

## 학습 계약으로의 매핑 (핵심 가정)
- 학습 r8 = 8마이크 링이 아니라 **binaural(L/R, ±y) × yaw 4방향(0/90/180/270°)**.
  각 채널의 명목 귀 방위각(L=yaw+90°, R=yaw−90°)에 가장 가까운 리얼 마이크를 배정하고,
  같은 방위를 두 번 방문하는 중복 슬롯에는 45° 오프셋 마이크를 배정(8마이크 전부 1회씩 사용).
- 직접음(최대 피크)을 학습과 동일하게 **49번째 샘플**에 정렬(클립별 트리밍으로 절대 동기는 원래 소실).
- 44.1→48kHz 폴리페이즈 리샘플, 2799샘플(왕복 10m) 절단, **yaw-페어 단위 피크 1.0 정규화**(학습 계약).
- 리얼 리그의 방위 0점·회전방향이 모델 프레임과 미정이므로 **출력 파노라마는 45° 배수 회전 + 미러까지만 정의**됨.
  L/R 스왑(미러) 변형도 함께 저장.

## 결과
- `pred_depth.npy` (256×512 ERP, m), `pred_depth_mirror.npy`, `pred_depth.png`(시각화)
- 예측 범위 0.21–4.17m, 평균 1.53m. 한 방향(~90° 폭)에 0.3–0.5m의 매우 가까운 표면,
  나머지 방위는 2–3m 벽, 한 방향 ~4m 개방부 → 폭 5–7m급 실내로 읽힘.
- 일관성 체크: L/R 스왑 입력이 레이아웃을 거의 정확히 좌우 반전시킴(구조 신호에 반응한다는 증거).
- 주의: Replica(시뮬레이션 IR) 학습 모델의 실측 도메인 제로샷이므로 절대 스케일·디테일은 보수적으로 해석.
  6m 이상 예측이 없는 것은 방이 실제로 작을 수도, 도메인 갭으로 원거리 단서가 약해진 탓일 수도 있음.

재현:
```
CUDA_VISIBLE_DEVICES=<gpu> python3 cw_realdata/infer_realdata.py
```

## MP3D 모델 추론 추가 (같은 날)
`--run 0820_eatllrd_r8novd_mp3d --data-module data_mp3d --tag _mp3d`
(MP3D 8ch 최고 시드, test 0.7306; WINDOW 2823 자동 적용)
- `pred_depth_mp3d.npy/.png`, `pred_depth_mirror_mp3d.npy`
- 예측 0.29-3.96m, 평균 1.65m. 근접면(0.3-0.5m)이 0-130도로 더 넓고, 최원방 3.9m@~255도,
  마주보는 축 스팬 3.1-4.1m.
- 두 모델의 수평선 프로파일 상관 r=0.53(회전/미러 보정 최적 r=0.57) — "가까운 면 하나 +
  2.5-4m 벽 + 한 방향 개방부"라는 방 구조 자체는 합의, 개방부의 방위는 서로 다르게 판정
  (Replica판 ~102도 vs MP3D판 ~255도; 도메인 갭 하 제로샷의 한계로 방위 배치는 참고 수준).

## batvision_corner 씬 추가 (같은 날)
`batvision_corner.zip` → `batvision_corner_x/` (clipped_audio 8ch + 원본 mono_audio + 스테레오
캘리브레이션 wav/json). `--audio-dir` 인자 추가로 같은 스크립트 재사용:
```
python3 cw_realdata/infer_realdata.py --audio-dir batvision_corner_x/batvision_corner/clipped_audio --tag _corner
python3 cw_realdata/infer_realdata.py --run 0820_eatllrd_r8novd_mp3d --data-module data_mp3d \
    --audio-dir batvision_corner_x/batvision_corner/clipped_audio --tag _corner_mp3d
```
- 결과: pred_depth_corner*.npy/.png
- 두 모델 모두 **코너 시그니처**를 출력: 한쪽 ~90-135도 호에 0.5-1.4m의 가까운 벽 두 면,
  반대 반구는 3.4-4.7m로 열림. Replica판 최근접 0.70m@139도/최원방 4.72m@255도,
  MP3D판 0.50m@49도/3.43m@269도(스케일은 MP3D판이 전반적으로 작게).
- 모델 간 수평선 상관 r=0.62 (첫 방 0.53보다 높음 — 코너처럼 비대칭이 강한 장면에서 합의가 좋아짐).
- 첫 방 예측과 코너 예측의 상관은 r=-0.15: 출력이 장면에 실제로 반응한다는 (고정 prior가 아니라는) 추가 증거.

## results/ — 모델 3종 × 데이터셋 2종 × 씬 2종 종합 (같은 날)
`results/pred_depth_{scene}_{dataset}_{model}.npy/.png` + `summary_{scene}.png`(그리드)
+ `summary_{scene}_horizon.png`(수평선 오버레이). 생성: `results_summary.py`.
- 모델: oaa(=캠페인 AFM: Replica판 sslam_llrd_r8vd / MP3D판 eatllrd_r8novd),
  bat(BatVision bat_r8_fin), eco(EchoDiffusion eco_r8_fin / eco_r8; 전용 env,
  `infer_realdata_eco.py` + `results/_wav48_*.npy` 사전정렬 파형 사용)
- 관찰: OAA-AFM 두 판이 가장 깨끗한 장면 구조. BatVision/Replica는 OAA와 레이아웃 상관
  r=0.46-0.47로 대체로 합의(스케일은 작게, med 0.4-1.1m), BatVision/MP3D는 얼룩짐.
  EchoDiffusion/Replica는 실측 입력에서 ~1.2m 평탄 체커 텍스처로 붕괴(min 0.89/max 1.76 —
  장면 반응 거의 없음), MP3D판도 노이즈성. 시뮬→실측 제로샷 강건성: OAA-AFM > BatVision ≫ EchoDiffusion.
