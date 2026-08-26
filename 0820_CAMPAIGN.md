# 0820 AFM 캠페인 — 실행 체계와 모델 전략

*hear360 (OAA / HEAR 360°) 릴리스 코드 위에서, coarse per-observation 인코더만 pretrained
Audio Foundation Model로 교체했을 때 논문 CNN을 이길 수 있는가.* (2026-08-20 ~ 진행 중)

## 1. 무엇을 바꾸고 무엇을 고정하는가

교체 대상은 **coarse 인코더 하나뿐**이다. fine CNN skip path, pose/ear/TF 임베딩,
alternating + ray-mic geometric attention, 디코더, masked-L1 로스, EMA 0.999, val-MAE 선택 —
전부 논문 그대로 둔다. 이렇게 해야 성능 차이가 "인코더 표현력" 하나로 귀속된다.

- 백본: ViT-B/16 계열 5종(audiomosaic·bat·eat·sslam·m2d) + m2d20ms(patch 128×2 시간슬라이스)
- patch embed는 task-specific 새로 학습, pretrained 본체는 0.1×LR full fine-tuning
- 입력 정규화는 log1p + per-sample std (dB 계열은 Stage-2에서 기각됨)
- 코드: `model/audio_backbones_0820.py`, `0820_train_oaa_afm.py` — 기존 파일은 건드리지 않고
  0820_ 접두어로 병렬 존재. `core/ckpt.py`의 빌드 훅으로 eval.py가 그대로 재빌드 가능.

## 2. 단계별 전략 (가설 → 판정 → 다음 가설)

| 단계 | 질문 | 결과 |
|---|---|---|
| 1 스크리닝 | 5개 백본, 동일 레시피, 4ch fb 양 데이터셋 | 전패. 단 Replica는 곡선이 CNN과 포개짐(인코더 비병목), MP3D post-norm 모델은 LR 스파이크 |
| 2 variant | 실패 원인별 가설 검증 | **LLRD 0.75+warmup8이 언락** (eat MP3D 0.8912<CNN 0.892). randinit +29%↓(사전학습 유효), conv-stem·dB 기각, 시드분산 0.012 |
| 3 확장 | 승자(eat_llrd)를 멀티시드 + 2/4/6/8ch 스케일링 | MP3D fb 3시드 전승(0.772±0.002 vs 0.785), r2 양 데이터셋 승, r6/r8 Replica 박빙 |
| 3c/3d (진행 중) | vdrop(mic drop)을 AFM에 양방향 ablation | r6/fb에 추가, r8에서 제거 — OAA는 r8에서 vdrop이 핵심이었으므로 |

원칙: **한 번에 한 가설**, 판정은 반드시 test(val→test 순위 뒤집힘 다수 관측), 박빙이면 멀티시드.

## 3. 지금까지의 핵심 그림 (test MAE)

| obs | Replica AFM/CNN | MP3D AFM/CNN |
|---|---|---|
| r2 | **0.2762** / 0.2894 | **0.9018** / 0.9084 |
| fb | 0.2609 / 0.2596 (≈) | **0.7717±0.002** / 0.785 |
| r6 | 0.2427 / **0.2384** | (학습 중) / 0.7502 |
| r8 | 0.2371 / 0.2368 (≈, RMSE는 승) | (학습 중) / 0.7467 |

해석: AudioSet 사전학습은 **관측이 빈약하거나(r2) 데이터가 어려울 때(MP3D)** 이긴다.
대역 분해로 원인 확정 — AFM은 mid/far(잔향 큐)에서 항상 우세, CNN은 near<3m 정밀도에서 우세.
채널이 늘수록 근거리 삼각측량이 CNN에 유리해져 이득이 상쇄된다. 즉 AFM이 사는 것은
"정밀도가 아니라 강건성". EchoDiffusion 대비는 전 채널 승 + eco의 r8 비단조는 pose-blind
채널 집계(channel-mean CIDE) 탓임을 chdrop 역효과 실험으로 확정.

## 4. 실행 인프라

- **free-GPU 디스패처** (`0820_queue_*.sh`): GPU 2–7을 폴링(memory<2000MB)해 잡을 순차 투입.
  로그 파일 존재로 idempotent — 재실행해도 완료/진행 런은 건너뜀. `SEEDS="0 1 2"`로 멀티시드 확장.
- **생존성**: 모든 학습은 `setsid nohup … </dev/null` — 부모 셸/세션과 완전 분리.
- **감시 루프**: 세션의 Monitor(로그 [done]/에러 감시) + 시간 wakeup 이중화. 런 완주 즉시
  해당 데이터셋 test eval을 빈 GPU에 자동 투입 (`comparison_0820/compare.json`, `mp3d_eval/`).
- **매 epoch last.pth** 저장 → `--resume`으로 중단 복구. 결과는 `0820_collect.py`로 일람.

## 5. 남은 일

1. MP3D r6/r8 + vdrop ablation 4런 완주·평가 → 채널 스케일링 표 완성
2. vdrop 판정: OAA처럼 이득이면 r6/fb 박빙 역전 시도의 근거, 아니면 ViT 특이성으로 기록
3. 최종 결론을 `0820_RESULTS.md`에 반영, 필요 시 박빙 셀만 멀티시드 추가
