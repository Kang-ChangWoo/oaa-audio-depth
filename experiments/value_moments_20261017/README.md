# Value-moment readout: 작은 구조 변경과 재현성 수정

기준: review snapshot `27f6555e19c347f1336fa2eeac2aa3941c0c0075`.
**구현·동작 검증 완료, Replica/MP3D 성능 향상은 미검증.**
이 snapshot에는 데이터, 학습 가중치, 원본 실험 결과가 없다. README의 실험 수치는
제공된 문서의 주장으로만 취급했다. 기본 모델은 유지하며 `--value-moments`로 켠다.

## 1. 코드에서 확인한 문제와 판단

| 확인한 내용 | 의미와 조치 |
|---|---|
| `train_oaa.py`가 PyTorch/NumPy만 seed하고 Python `random`을 seed하지 않음 | observation dropout과 yaw augmentation이 동일 seed에서도 달라짐. `random.seed(a.seed)` 추가. **새 대조군도 다시 실행해야 함.** |
| 실제 STFT 설정은 manifest에만 있고 checkpoint args에는 없음 | `best.pth`만 옮기면 hop-16을 hop-160으로 해석할 수 있음. hop/window/FFT/crop을 args에도 저장. 기존 manifest fallback은 유지. |
| `ray_gate` 입력인 `q_ln`은 audio를 보기 전의 고정 ray query | sample별 음향 증거의 유무를 직접 판단할 수 없음. pose-only GeoFiLM도 주어진 pose에서 sample에 따라 바뀌지 않음. |
| `RayMicAttn` readout이 attention-weighted value 평균 하나만 전달 | 평균이 같고 분산이 다른 token 집합은 이 readout에서 구분되지 않음. 분산 정보를 추가하는 후보 구현. |
| `quick_val`은 전체 픽셀을 모으고, 최종 headline metric은 이미지별 평균 | 선택 기준과 보고 기준이 다름. 기존 프로토콜을 바꾸지 않고 probe에서 둘 다 보고. |

**softmax 가중치 합이 1이라는 사실은 출력 norm이 고정된다는 뜻이 아니다.**
예를 들어 value가 `+1, -1`이면 가중치 `0.5, 0.5`의 출력은 0이고,
`0.9, 0.1`의 출력은 0.8이다. 기하 bias로 가중치를 바꾸는 것만으로도 크기가 변한다.
README의 거의 일정한 norm 측정만으로 softmax가 성능 정체의 원인이라고 확정할 수 없다.
이 패치는 그 설명을 전제로 gate를 추가하지 않고, 평균으로 압축하면서 소실되는
또 다른 통계량이 유용한지를 검사한다. 분산이 mid-band 깊이의 증거라는 보장은 없다.

## 2. 변경한 구조

ray/head별 기존 attention 확률 `p`와 value `V`에 대해:

```text
mu    = sum_k p_k V_k                       # 기존 readout 그대로
var   = E_p[(V - a)^2] - E_p[V - a]^2        # a: 유효 token 하나, 수치적 centring용
sigma = sqrt(max(var, 0) + 1e-6) - 1e-3
out   = mu + gamma * sigma                  # 이후 기존 output projection, residual, FFN
```

- `gamma`는 head/channel별 0 초기화 파라미터. C=256에서 **정확히 +256개**.
- `sigma`는 실제 음향 value 분포와 geometry-biased attention으로부터 계산한다.
  LSE, attention temperature, loss 변경, 추론 후 depth 보정은 사용하지 않는다.
- 기존 mean만 사용하는 readout과 달리, 같은 평균의 분포를 구분할 수 있다.
  `gamma`가 학습되면 출력은 기존 value들의 convex combination으로 제한되지 않는다.
- 초기 출력과 공유 파라미터 gradient는 CPU float64에서 기존과 정확히 같고,
  `gamma`의 gradient는 처음부터 존재한다. 초기화는 RNG도 소비하지 않는다.
- 분산 계산은 mixed precision에서도 FP32. float64는 유지한다.
  mask가 있으면 centring 기준도 유효 token에서 골라 masked 입력의 영향을 배제한다.
- 실험 해석을 위해 global acoustic softmax에서만 지원한다.
  registers / GeoFiLM / film-lse / ray-gate / no-cross와 섞으면 명시적으로 실패한다.
- **파라미터 비용과 연산 비용은 다르다.** FP32 value matmul 2개와 중간 tensor가 추가된다.
  GPU 시간·VRAM 증가량은 아래 smoke로 측정해야 한다. CUDA 비용은 여기서 측정하지 못했다.

## 3. 변경 파일

| 파일 | 역할 |
|---|---|
| `model/oaa.py` | value dispersion readout, masking, 진단 통계 |
| `train_oaa.py` | 새 flag, Python seed, checkpoint STFT 기록, DDP world size 기록, parser 분리 |
| `core/ckpt.py` | checkpoint args에서 새 모듈 복원; 과거 args 기본값은 off |
| `tests/test_oaa_value_moments.py` | identity, gradient, masking, BF16, moment 통계, strict load |
| `tests/test_value_moment_experiment.py` | paired recipe 보존, 설정 누락 차단, overwrite 방지 |
| `tests/gpu_smoke_attn_fusion.py` | reference vs moments GPU 비용 측정 옵션 |
| `probe.py` | 기존 가중치를 고정하고 gamma만 train subset에서 학습하는 짧은 진단 |
| `run_pair.py` | 실제 control manifest에서 설정을 복사한 from-scratch 대조 실험 |

## 4. 적용 및 동작 검증

패치를 받은 경우 **review snapshot checkout**에서 적용한다. 원본 live repository는
working-tree 변경이 더 있을 수 있으므로 먼저 `git apply --check`로 충돌을 확인한다.
진행 중인 학습이 읽는 checkout 대신 별도 checkout/branch에서 적용한다.

```bash
git show HEAD:model/oaa.py > /tmp/oaa_before_value_moments.py
git apply --check /path/to/oaa_value_moments.patch
git apply /path/to/oaa_value_moments.patch
OAA_MOMENT_REF_MODULE=/tmp/oaa_before_value_moments.py python tests/test_oaa_value_moments.py
python tests/test_value_moment_experiment.py
```

로컬 검증 환경: PyTorch 2.8.0+cpu / torchvision 0.23.0+cpu, CUDA 없음.
**11개 model gate + 5개 experiment gate 통과.** 여기에는 N=2/4/8의 full-model 초기 출력
차이 0, RayMicAttn 공유 gradient 차이 0, live adapter gradient, trained-mask 불변성,
CPU BF16 forward/backward, checkpoint 재구성, production-width +256개 확인이 포함된다.
합성 spectrogram cache와 임시 checkpoint로 probe의 데이터 로드 → adapter 학습 →
validation 집계 → 저장까지 실행했다. 이 합성 실행의 수치는 성능 근거가 아니다.
실제 학습 checkpoint의 parity, CUDA 수치 안정성 및 GPU 비용은 아직 미검증이다.

## 5. GPU 비용부터 확인

결과는 repository 밖에 저장한다. 아래의 경로는 예시 placeholder다.

```bash
python tests/gpu_smoke_attn_fusion.py --value-moments-only \
  --nviews 8 --batch 3 --stem-stride1 --json /path/to/results/moment_gpu_smoke.json
```

무리 없이 맞는 micro-batch인지 확인하고, reference 대비 VRAM과 학습 step 시간을 보고
전체 실험 비용을 판단한다. 기존 campaign의 stem 설정이 다르면 그 설정에 맞춘다.

## 6. 먼저 짧게 검사: frozen-backbone probe

`CONTROL_RUN`은 최신 **hop-16 control의 best.pth와 manifest.json이 있는 run 디렉터리**다.
현재 데이터 root와 split 환경을 먼저 설정한다. 과거 `_fin` 설정과 섞지 않는다.

```bash
source env.local.sh
python experiments/value_moments_20261017/probe.py \
  --baseline-run /path/to/control_run \
  --out-dir /path/to/results/moment_probe_replica \
  --train-samples 256 --val-samples 256 --passes 2 --batch-size 2
```

이 스크립트는 train에서만 gamma를 학습하며, validation은 전후 측정만 한다.
checkpoint의 STFT를 복원하고, train/val scene 중복을 거절하며,
scene별 균형을 맞춘 고정 subset의 index와 before/after metric을 기록한다.
전체 weighted MAE, pooled selection MAE, near/mid/far MAE, scene별 수치를 모두 본다.
추가 모델 선택이나 test 조회는 하지 않는다. `adapter.pth`는 probe용 gamma만 담으며
최종 평가용 checkpoint가 아니다.

이 probe는 LR을 원래 control에서 가져오지만, backbone 고정·짧은 일정·EMA 미사용이라는
차이가 있다. 따라서 **이득이 있어도 최종 성능 향상의 증거가 아니고, 이득이 없어도
joint training에서 쓸모없다는 증명은 아니다.** 효과가 거의 없으면 비싼 확인 실험의
우선순위를 낮추는 용도다. 추가 random subset을 반복해 좋은 결과만 고르지 않는다.

## 7. 확인 실험: Replica N=2와 MP3D N=8, 각각 paired seeds

```bash
python experiments/value_moments_20261017/run_pair.py \
  --baseline-run /path/to/control_run \
  --out-root /path/to/results/moment_pairs_replica \
  --baseline-world-size 1 --seeds 0 1 2
```

기본은 실제 실행 없이 명령만 출력한다. 확인 후 같은 명령에 `--execute`를 추가하면
control/moments를 각 seed별로 순차 실행한다. MP3D N=8 control에 대해서도 별도 실행한다.
이전 manifest에 world size가 없으면 실제 run을 확인하고 `--baseline-world-size 1`을 준다.
DDP control을 single GPU로 바꾸는 것은 허용하지 않는다. 원래 환경의 데이터 root/split을
유지해야 한다; 옛 manifest에는 모든 split 환경 변수가 저장되어 있지 않다.

- epoch horizon, warmup, batch/accumulation, dropout, stem, FFT/window/hop/crop은 동일하게 유지.
- loss는 기존 plain masked L1, aux L2=0. 추가 모듈 외 recipe 변경은 하지 않음.
- Python seed 수정은 두 arm에 동일 적용. 과거 unseeded 숫자를 control로 빌려오지 않음.
- 기존 run 디렉터리 덮어쓰기, 모르는 architecture flag, STFT 기록 누락은 오류 처리.
- 이번 driver는 **fresh runs만** 지원한다. 기존 trainer의 resume는 RNG 복원과 best/history
  보존에 별도 문제가 있으므로, 중단 후 resume를 같은 재현성 실험으로 취급하지 않는다.
- validation으로 비교·선택하고, 최종 고정 후에만 기존 `eval.py`로 test를 평가한다.

최종 판단은 두 dataset/config 모두에서 paired seed별 MAE/RMSE/mid MAE 차이를 보고 한다.
3개 seed 평균 차이뿐 아니라 개별 차이와 paired confidence interval을 기록한다.
README의 3.99%는 기존 control 변동성에 관한 참고값이며, 새로운 paired comparison의
통계 검정을 대체하지 않는다. 한 seed의 작은 이득이나 image 수를 seed 수로 대체한
유의성만으로 성공 판정을 하지 않는다. near-band 손해와 전체 MAE 악화도 함께 확인한다.

이 후보로도 mid-band 이득이 없으면, 더 많은 gate를 붙이기보다 실제 checkpoint에서
음향 shuffle/mean-depth prior 대비 추가 정보량과 3–6 m 오차를 다시 측정하는 편이 낫다.
현재 snapshot만으로 mid-band 정체의 지배적인 원인을 확정할 수는 없다.
