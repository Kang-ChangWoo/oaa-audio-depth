# cw_realdata — 리얼 녹음 추론 파이프라인 (synthetic→real zero-shot)

## 폴더 구조
```
data/<scene>/               # 원본 녹음 (room / corner / hall) + zip 보관
  clipped_audio/            #   mono_{45..360}deg_clip_1s.wav (44.1kHz, 8마이크 45° 링)
  batvision_XX_YY/          #   180° 마주보는 마이크 페어의 스테레오 캘리브레이션 (corner/hall)
results/<scene>/            # 추론 결과: pred_{replica|mp3d}_{oaa|bat|eco}.npy/.png
  summary.png               #   모델 3종 × 데이터셋 2종 그리드
  summary_horizon.png       #   수평선 프로파일 오버레이
  extra/                    #   mirror 변형, 정렬된 파형(_wav48.npy), 구버전 산출물
results/audit/              # 입력 감사 산출물 (input_audit.png, TDR 보정 실험)
infer_realdata.py           # OAA-AFM/BatVision/EchoScan 추론 (--scene, --run, --data-module, --match-tdr)
infer_realdata_eco.py       # EchoDiffusion 추론 ($ECHODIFF_PY 전용 env, extra/_wav48.npy 사용)
results_summary.py          # summary 그림/수치 재생성
input_audit.py              # sim-to-real 입력 감사
```
모델: oaa = 캠페인 최고 8ch(Replica판 `0820_sslam_llrd_r8vd_rep` 0.2363 / MP3D판
`0820_eatllrd_r8novd_mp3d` 0.7306), bat = BatVision `bat_r8_fin`, eco = EchoDiffusion
`eco_r8_fin`/`eco_r8`.

## 실행
```
python3 cw_realdata/infer_realdata.py --scene room --tag _replica_oaa                       # 기본(Replica OAA)
python3 cw_realdata/infer_realdata.py --scene hall --run comparison/bat_r8_fin --data-module data_0422 --tag _replica_bat
$ECHODIFF_PY cw_realdata/infer_realdata_eco.py --run comparison/eco_r8_fin --data-module data_0422 \
    --wav48 results/room/extra/_wav48.npy --out-dir results/room --tag replica_eco
python3 cw_realdata/results_summary.py && python3 cw_realdata/input_audit.py
```

## 학습 계약으로의 매핑 (가정 전부 infer_realdata.py docstring에)
- 학습 r8 = binaural(L/R=yaw±90°) × yaw 4방향. 8마이크 링을 명목 귀 방위각 최근접으로 배정
  — 리얼 리그의 180° 스테레오 페어(45/225 등)가 모델의 yaw-페어와 정확히 일치.
- 직접음을 학습 계약(49번째 샘플)에 정렬. **정렬 기준은 전역 피크가 아니라
  onset(>50% max 첫 교차 후 1ms 내 국소 피크)** — hall/215°에서 늦은 반사음(25ms)이
  직접음(3.6ms)보다 커서 argmax 정렬이 -21ms 오염을 만드는 사례를 발견해 수정(2026-09-08).
- 44.1→48kHz 폴리페이즈, WINDOW(2799/2823) 절단, yaw-페어 단위 피크 1.0 정규화(학습 계약 검증됨).
- 출력 파노라마는 45° 배수 회전 + 미러까지만 정의(리얼 리그 0점/손방향 미상).

## 입력 감사 결과 (input_audit.py, results/audit/input_audit.png)
| 항목 | REAL (24ch) | TRAIN sim | 판정 |
|---|---|---|---|
| tail/direct 에너지비 | **10.0** (2.9–27.9) | **1.1** (0.2–4.3) | ⚠️ 최대 갭 — real 직접음이 상대적으로 ~9배 약함 |
| direct 폭 | 0.20ms (–0.40) | 0.07ms | 스피커/마이크 대역 제한, ~3배 넓음 |
| 노이즈 플로어 | 0.042 | 0.002 | ~20배 (관측창이 58ms라 영향 제한적) |
| 페어 내 도착차 | −0.36~−0.75ms 일관 | 両귀 0 (계약) | per-채널 정렬이 계약에 부합 |
| hall 45/215 | −21ms 이상치 | — | 원인=argmax 정렬 버그 → onset 정렬로 해결 |

**TDR 보정 실험** (`--match-tdr 1.1`: tail을 감쇠해 sim 비율로): 레이아웃은 거의 불변
(r=0.94–0.97), 절대 거리만 +0.4~0.5m 이동(med 2.6→3.0m). 결론: **방위 레이아웃 판독은
진폭 도메인 갭에 강건하고, 절대 스케일에는 ±0.5m급 계통 편향 가능** — 실측 해석 시
uncorrected와 tdr-corrected 사이를 신뢰구간으로 볼 것.

## 씬별 결과 요약 (canonical, oaa 기준)
- **room**: 한 방향 0.3–0.5m 근접면 + 2.5–3m 벽 + 최원방 4.1m. 방 스팬 ~3.3×5.5m.
- **corner**: 90–135° 호에 0.7–1.4m 인접 벽 2면(코너 시그니처) + 반대 반구 3.8–4.7m.
- **hall**: 최근접 1.0m/최원방 4.3–5.1m, 모델 간 합의 가장 낮음(r≈0.49) — 10m 관측창을
  넘는 공간일 가능성.
- 모델 강건성 순위(제로샷): **OAA-AFM > BatVision ≫ EchoDiffusion**(Replica판 eco는 전 씬에서
  ~1m 평탄 텍스처로 붕괴, 장면 반응 없음).

## 전처리 권고 (2026-09-09 그리드 실험, results/audit/pred_*_{rep,mp}_{none,hp,tdr,hptdr}.npy)
GT가 없으므로 "독립 학습된 두 모델(Replica판 vs MP3D판)의 수평선 합의도"를 기준으로
{없음, 하이패스 200Hz, TDR 보정, 둘 다}를 3씬에서 비교했다.
- **권고 = 현행 최소 처리 유지**(onset→49 정렬 + 페어 피크 정규화 + 계약 STFT). 평균 합의 r=0.50으로 최고.
- **하이패스 기각**: 저주파 웨이브를 제거하자 예측 스케일이 붕괴(중앙값 2.6→0.5-0.7m).
  모델의 거리 판독이 저주파 잔향 에너지에 크게 의존한다는 뜻 — sim에도 LF 밴드가 있어
  "sim에 없는 성분"이라는 가정이 틀렸다.
- **TDR 보정은 기본값으로 채택하지 않음**: hall에서는 합의를 0.49→0.80으로 올리지만
  corner에서는 -0.66으로 불안정. 절대 스케일의 불확실성 구간(±0.5m) 추정용으로만 사용.

## 논문 체크포인트 × 전처리 3종 전수 비교 (results/paper/<scene>/grid.png)
논문 최종 pt(oaa_r8_fin, bat_r8_fin, eco_r8_fin/eco_r8) × {현행, 하이패스200Hz, TDR} × 3씬.
- **논문 모델들에서는 하이패스가 명확한 승자**: OAA Rep↔MP3D판 합의 r가 none 0.02/0.31/-0.63
  → hp 0.74/0.57/0.60 (room/corner/hall). 육안으로도 OAA·BatVision의 MP3D판에서 none의
  포화 아티팩트(>6m 붉은 덩어리)가 hp에서 사라지고 방 구조가 정리됨.
- AFM(0820) 모델들은 반대로 hp에서 스케일이 줄었음(위 절 참조) — 전처리 효과가 모델 계열
  의존적. 해석: 논문 CNN/BatVision은 실측 저주파 웨이브를 노이즈로 오독해 아티팩트를 만들고,
  sslam 계열은 저주파 잔향 에너지를 거리 단서로 사용.
- 실무 권고(수정): 논문 계열 모델로 실측 추론 시 hp200 적용, AFM 계열은 현행 유지 + 두 판 병렬
  확인. 최종 확정은 실측 GT 대조가 필요.
