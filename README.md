# OAA: Orientation-Aligned Attention for Audio-only ERP Depth Estimation

Binaural(양귀) 오디오 + 장치 방향(yaw) 정보만으로 ERP(equirectangular) depth 맵(256×512, 최대 10 m)을
추정한다. 이 폴더는 **정리된 baseline 참고 구현**이다 — 연구 리포(상위 폴더)의 실험 기능을 전부 걷어내고,
검증된 최종 레시피만 남겼다. 기존 학습 체크포인트와 **state-dict 완전 호환**(strict load + 출력 비트 일치 검증됨).

## 결과 (Matterport3D test 3600, MAE는 m 단위, 낮을수록 좋음)

| 입력 채널 | BatVision (54.5M) | OAA (11.0M) |
|---|---|---|
| 2ch (0° 양귀) | 0.920 | 0.919 |
| 4ch (cB) | 0.852 | 0.783 |
| 6ch (0/90/270° 양귀) | 0.808 | 0.738 |
| **8ch (4방향 양귀)** | 0.804 | **0.718** ← SOTA |

- 2ch에서는 두 모델이 동률 — OAA의 우위는 아키텍처 일반이 아니라 **다중 방향 채널을 pose-조건
  attention으로 융합하는 능력**에서 나온다 (pose 라벨을 셔플하면 성능 붕괴로 검증됨).
- BatVision(채널 스택 CNN)은 6ch에서 포화하지만 OAA는 8ch까지 계속 이득.

## 파일

| 파일 | 내용 |
|---|---|
| `oaa.py` | OAA 모델 (`OAAv2Depth`). view별 가중치 공유 인코더 → AdaLN pose 조건화 → intra/inter-mic 교대 attention → ray-mic 기하 bias cross-attention → ERP self-attention → 업샘플 디코더 |
| `batvision.py` | 비교용 BatVision U-Net (`RotDepth`, IROS 2023 재현) |
| `data.py` | mmap 캐시 로더 (모드 r2/cB/r6/r8; 캐시 포맷·STFT 레시피는 파일 docstring 참조) |
| `train_oaa.py` | OAA 학습 (champion 레시피 고정) |
| `train_batvision.py` | BatVision 학습 |
| `eval.py` | test 평가 (cos-latitude 가중, per-image 배치 불변; MAE/RMSE/AbsRel/δ1 + near/mid/far) |

## 실행

```bash
# OAA 8ch (SOTA 레시피)
python3 train_oaa.py --run-name oaa_r8_s0 --nviews 8

# BatVision 8ch
python3 train_batvision.py --run-name bat_r8_s0 --mode r8

# 평가 (모델 종류는 ckpt에서 자동 인식)
python3 eval.py --run-name oaa_r8_s0 bat_r8_s0
```

## 모델 핵심 (oaa.py)

1. **View별 독립 인코딩** — 각 orientation-ear 채널을 같은 CNN으로 따로 인코딩 (배치축으로 펼침).
   파라미터는 nviews 불변(11.0M) → 채널 추가 비용이 0.
2. **결정적 pose 기하** — 각 view의 (yaw, ear)를 알고 있다는 전제. pose 임베딩(additive)과
   **AdaLN**(LayerNorm 변조, 이 프로젝트 유일의 아키텍처 이득 −0.02)으로 주입하고,
   `RayMicAttn`이 출력 ray별로 "ray를 그 마이크 좌표계에서 본 방향 + ear 축 정렬도"를 attention bias로 사용.
3. **intra/inter 교대** — view 내부 self-attention(AdaLN 조건) ↔ view 간 attention을 2라운드 교대.
4. 출력: coarse 16×32 토큰 → conv 업샘플 → sigmoid × 10 m (radial depth).

`view_poses` 인자로 임의 (yaw, ear) 조합을 넘기면 **학습 없이 임의 subset에 zero-shot 일반화**된다
(검증됨: 임의 4-of-8 subset에서 재학습 모델과 동일 성능).

## 학습 레시피 — 전부 검증으로 확정된 것

- **masked L1 하나만.** silog/berhu/depth-bins/uncertainty/aux 손실 전부 3-seed test에서 무이득~유해로 기각.
- **AdamW lr 1e-3 + warmup 4ep + cosine (30ep, bs32)** — CNN용 2e-3은 transformer를 발산성 정체시킴
  (이 LR 수정이 champion의 결정적 요소). **주의**: nviews=8에서는 1e-3도 경계값이라 시드 3개 중 1개가
  warmup 직후 loss 반등 후 미회복으로 실패함 — 그 패턴이 보이면 재시작하거나 `--lr 5e-4`.
- **EMA 0.999** 가중치를 저장/평가. bf16 autocast, grad clip 1.0.
- 모델 선택은 val(cos-lat 가중 MAE), 보고는 test. val이 test보다 ~0.11 높음. **시드 노이즈 ±0.018 —
  단일 시드로 우열 주장 금지.**

## 데이터 요구사항

`data.py` 상단 경로 상수 2개를 환경에 맞게 수정. 필요한 캐시(빌드 방법은 `data.py` docstring):
0° magnitude STFT + depth/mask (`RAD`), +90/180/270° STFT (`REAL90`). 같은 위치에서 4개 yaw의
**실제 binaural 녹음**이 있어야 6/8ch 모드가 성립한다.

## 알려진 확장 (연구 리포에서 검증, 이 baseline에는 미포함)

- **IPD 입력** (view당 [mag, cosIPD, sinIPD] 3채널): 4ch에서 0.783→0.763 (+검증 진행 중). 위상이 유일하게
  남아있던 추가 정보였음. 반대로 GCC-PHAT를 별도 브랜치로 주입하는 형태는 실패 — 정보는 같아도
  **per-view 입력 채널 형태**여야 attention이 활용함.
- **채널 정체성 추론**: 어느 귀(L/R)인지는 오디오만으로 ~100% 판별 가능(HRTF 비대칭), yaw는 판별 불가
  → pose(yaw)는 장치가 알아야 함(IMU 등), L/R 배선은 몰라도 됨.
- 실패 목록(재시도 비추천): Fourier PE, azimuth 증강, 해상도 증가, matched-echo 브랜치, subset 특화 학습,
  절대 위상(complex) 입력, Mid/Side 변환. 근거는 연구 리포 `analysis/COLD_REVIEW.md`.
