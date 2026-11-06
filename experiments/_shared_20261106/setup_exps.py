"""Index every 2026-11-06 request into its OWN self-contained experiment dir under the repo.

Each dir gets: README.md (question, design, cells, control, decision rule, current numbers,
reproduce command, limits), scripts/ (its own copies, no cross-dir imports), results/ (.gitignore'd,
filled by sync_experiments.sh), and the PREREG where one exists.
"""
import os
import shutil

REPO = "/root/storage/implementation/shared_audio/hear360"
EXP = os.path.join(REPO, "experiments")
OLD = os.path.join(EXP, "20261106_replica_pose_sharing")
POSE = "/root/local1/changwoo/poseablate_20261106"
SEED = "/root/local1/changwoo/relevel_seedfloor_20261016"

ABL_HOWTO = """## 재현 (학습 0, 채점만)
```
cd /root/storage/implementation/shared_audio/hear360
export CUDA_VISIBLE_DEVICES=<카드> EVAL_BS=4 ABL_MODES=<모드,모드,...>
/opt/conda/envs/shared_audio/bin/python scripts/pose_ablate2.py replica \\
  <run_dir>,<run_dir>,... results/<out>.json
```
`pose_ablate2.py` 는 체크포인트에서 포즈를 복원하고 채점할 때 리스트만 바꿔 끼운다. 같은 json 에
이어쓰므로 중간에 끊겨도 손해가 없고, 이미 있는 모드는 건너뛴다. 지표 11종 전부 저장한다.

**검산**: `true` 모드는 그 체크포인트의 기록된 test AbsRel 과 일치해야 한다. 지금까지 3/3 일치
(`paperrep_ctrl_fb_s1` .14360, `ctrl_fb_s2` .14962, `ctrl_r8_s4` .20713).

**2관측에서 무효인 모드**: yaw 가 둘 다 0 이므로 `randyaw*`·`pairbreak` 는 항등이고
`randear*`·`earflip*` 는 `swap` 과 같아진다. 스크립트가 `_identity_noop` 으로 표시한다.
"""

LIMITS_ABL = """## 한계
- 각 칸 1시드가 아니라 가능한 시드를 전부 돌리지만, 8관측은 체크포인트가 ctrl·ILD 각 2시드뿐이다.
- 포즈 교란은 +200 % 이상 포화 구간이라 팔 간(기본 입력 대 ILD) 차이는 읽으면 안 된다.
- 비율 지표(AbsRel)가 가장 크게 움직이고 RMSE 는 완만하다. 밴드로 쪼개 해석하지 말 것
  (주지표는 전체 AbsRel).
"""

READMES = {}

READMES["20261106_unshared_encoder"] = f"""# 가중치 공유를 끊은 인코더 (Replica 2·4·8관측)

창우 요청 2026-11-06 19:13: "원래 코드가 weight sharing 한 encoder 들인 거잖아. 근데, 그게
아니라 weight를 나누지 않은 애들을 하자는거야". 19:48: "이 실험들은 주로 replica에서".

## 질문
레퍼런스 모델은 모든 (yaw, 귀) 관측을 **하나의 공유 인코더**로 인코딩한다. 그 공유가 성능을
떠받치고 있는가, 아니면 그냥 파라미터 절약인가.

## 무엇을 바꿨나 (`--unshared-enc`, 커밋 16a62ba)
`ViewEncoder` 아키텍처 그대로 `nviews` 벌을 세우고 관측 i 를 인코더 i 가 맡는다. 관측 0 은
`enc.*` 키를 유지하고 나머지는 `enc_extra.*` 로 **스위치가 켜질 때만 생성**되므로, 꺼진 상태에서는
레퍼런스 체크포인트가 strict 로 그대로 로드된다. 토큰 4,096 · 포즈 8개 · 관측 내/간 어텐션 ·
레이 경로 · 디코더 · 입력 텐서 · 레시피 전부 ctrl 과 동일하다. 인코더는 639만 파라미터(모델의
41.6 %)이므로 전체가 2관측 ×1.416, 4관측 ×2.249, 8관측 ×3.915 로 커진다.

## 칸과 대조군 (주지표 test AbsRel, 모집단은 처음부터 고정)
| 칸 | 대조군(착지) | 노드 | 런당 |
|---|---|---|---|
| `paperrep_unshared_r2_s0..s4` | `paperrep_ctrl_r2_s0..s4` .16102 (sd .00403) | n14 | 181 s/ep, 2.0 h |
| `paperrep_unshared_fb_s1..s4` | `paperrep_ctrl_fb_s1..s4` .146625 | n1 | 측정 예정 |
| `paperrep_unshared_r8_s0..s4` | `paperrep_ctrl_r8_s0..s4` .14899 (sd .03283) | n1 | 758 s/ep, 8.4 h |

## 판정 기준 (사전등록, 사후 변경 금지)
`PREREG_unsharedenc_20261106.md` 참조. 칸마다 자기 바에서만 판정하고 다른 칸의 유의성을 빌리지
않는다. 2관측이 용량 교란이 가장 작아(+41.6 %) 가장 깨끗한 테스트다.

## 스모크 (전부 통과)
스위치 OFF 가 패치 전 트리와 **18개 경우 비트 동일**(관측/채널 수 8가지 × 드롭 유무 + 실제
256×512 해상도 2개), 인코더 3만 흔들면 관측 3 토큰만 변하고 나머지 7개는 **정확히 0.0**,
양방향 strict 로드, `core/ckpt.py` 가 플래그를 넘겨 채점·재개가 같은 구조를 세움, AdamW 1스텝 정상.
재현: `python scripts/smoke_unsharedenc.py out.json --full`.

## 한계
파라미터가 커지는 교란은 **창우 결정(20:20)으로 실험으로 가르지 않는다**. 결과를 쓸 때 같은
문장에서 용량 배수를 밝히고, 2관측 칸을 앞세운다.

## 파일
`PREREG_unsharedenc_20261106.md`, `scripts/patch_unsharedenc.py`(코드 수정),
`scripts/smoke_unsharedenc.py`, `results/`(학습 로그와 STATUS).
"""

READMES["20261106_pose_random_heading"] = f"""# 랜덤 헤딩 포즈 교란 (Replica, 학습 0)

창우 요청 2026-11-06 19:48: "랜덤헤딩 주는 실험도 마찬가지(Replica에서)". Gibson 에만 있던
인퍼런스 전용 포즈 교란의 Replica 판이다.

## 질문
모델이 관측별 포즈(헤딩)를 실제로 쓰는가. 쓴다면 무엇에 민감한가 — 각도 값인가, 배정의
일관성인가, 아니면 전역 방향인가.

## 모드
`randyaw`·`randyaw2`·`randyaw3` yaw 를 채널 간 순열(시드 1·2·6) / `yawblind` yaw 전부 0 /
`yawroll` 전역 +90도 / `yawrot45`·`yawrot90`·`yawrot135`·`yawrot180` 전역 회전 스윕.

## 판정 [사실] — 통과
4관측 ctrl 4시드 평균: `randyaw` +241 %, `randyaw2` +328 %, `randyaw3` +567 %,
`yawblind` +254 %, `yawroll` +340 %. 사전등록 기준("순열 3개 전부 악화 + 평균 ≥10 %")을 4시드
전부 통과하고 최소값이 +225 %다. Gibson 같은 칸(+294 % 평균, 1시드)보다 Replica 가 더 민감하다.
ILD 팔도 +244 / +361 / +581 % 로 ctrl 과 구분되지 않는다(포화).

## 제일 중요한 줄 [사실 + 추론]
`randyaw3` 의 포즈 리스트를 꺼내보면 **짝 구조가 멀쩡하고 두 헤딩만 통째로 맞바꾼 것**이다
(= 전역 180도 회전). 그게 가장 나쁘다. 전역 90도 +340 %, 전역 180도 +567 % 로 회전량에 따라
커지고, 정보를 지우는 쪽(`yawblind` +254 %, `blind` +240 %)은 그보다 싸다.
→ 손해를 지배하는 것은 짝 구조의 붕괴가 아니라 **암시되는 전역 방향이 참값에서 얼마나
돌아갔는가**다 [추론]. 회전 스윕으로 확인 중이며, `yawrot180` 의 포즈가 `randyaw3` 와 같고
`yawrot90` 이 `yawroll` 과 같으므로 그 두 쌍이 같은 숫자를 내면 구현 검산까지 된다.

{ABL_HOWTO}
{LIMITS_ABL}
"""

READMES["20261106_ear_mixing"] = f"""# 귀 방향을 거짓으로 주는 교란 (Replica, 학습 0)

창우 요청 2026-11-06 20:09: "양쪽 귀가 90 270 도가 아니라 섞여서 inference 하는 것도 학습없이
테스트해보긴 해야될 거 같아".

## 배경 — 귀 방향은 어디에 들어 있나
포즈는 `[sin yaw, cos yaw, ear]` 이고 yaw 는 **머리 헤딩**, 귀는 ±1 라벨이다. 귀 방향 자체는
기하 바이어스의 두 번째 열 `r·(R(yaw)x̂)·ear` 로 들어간다(`model/oaa.py` 757~772행). 즉 귀 축은
헤딩을 90도 돌린 축이고, 왼귀는 헤딩−90도 · 오른귀는 +90도를 본다.

## 모드
`swap` 귀 부호 전부 뒤집기(90↔270) / `earblind` 전부 오른귀 / `earleft` 전부 왼귀 /
`randear`·`randear2`·`randear3` 귀 부호를 채널 간 순열(항등 배열은 거부) /
`earflip{{k}}` 뒤쪽 헤딩 k 개만 뒤집기 / `blind` 포즈 전부 동일.

## 판정 [사실]
4관측 ctrl 4시드 평균: `swap` +320 %, `blind` +240 %, `earflip1` +146 %, `randear` +146 %,
`randear2`·`randear3` +111 %, `earleft` +132 %, `earblind` +131 %.
2관측 ctrl s0: 원본 .15886 → `swap` .60707 (+282 %), `earleft` .34837 (+119 %),
`blind` .33454 (+111 %).

**섞는 것은 지우는 것과 비슷하고(+111~146 %), 일관되게 거짓인 귀는 2~3배 나쁘다(+282~320 %).**
헤딩에서 본 "모순 > 부재"와 같은 결이다.

## 구현에서 한 실수 (기록)
처음 쓴 `randear` 시드가 항등 배열을 뽑아 무효 칸이 나왔다(4관측 귀 배열은 6가지라 1/6 확률).
항등 거부를 넣고 json 을 지우고 다시 돌렸다. 무효 줄은 로그에만 남아 있다.

{ABL_HOWTO}
{LIMITS_ABL}
"""

READMES["20261106_channel_directions"] = f"""# 채널마다 다른 방향 — 귀 형태를 깨는 교란 (Replica, 학습 0)

창우 요청 2026-11-06 20:20: "8개 채널을 각각 다른 방향으로 귀의 형태를 유지하지 않도록 하는
테스트를 넣어". 20:34: "한쪽 방향에서 한 귀는 45도 다른 귀는 yaw 기준 270도 인 상황도 생기는거지".

## 모드와 왜 그 모드인가
- `spread` / `spreadear` — n 채널에 2π/n 간격 방향을 **증가 순서로** 하나씩. 귀 부호를 상수로
  박거나(spread) 원본 유지(spreadear). **주의**: 8관측 참 헤딩이 0·0·90·90·180·180·270·270
  이라 이 격자는 짝수 채널에서 참값과 일치해버린다. 그래서 `spreadear` 는 "절반만 45도 틀어진"
  약한 교란이다.
- `spreadoff` / `spreadoffear` — 반 칸 옮긴 격자(22.5도, 67.5도, ...). 참 헤딩과 하나도 겹치지
  않는다.
- `spreadperm`·`spreadperm2` / `spreadpermear`·`spreadpermear2` — 그 방향들을 **무작위 순서로**
  배정(증가 순서는 거부). 짝의 두 귀가 무관한 yaw 를 받는다: 예 (292.5, 157.5), (337.5, 112.5).
  창우가 말한 상황이 여기서 생긴다. 앞의 격자 모드들은 채널의 방위각 순서를 보존하므로 이게
  필요하다.
- `pairbreak` — yaw 열을 한 칸 굴린다. 짝은 깨지지만 yaw 값은 **학습에서 본 집합 안**에 남으므로,
  45도 같은 미지의 각도라는 교란이 없는 버전이다.

## 지금까지 [사실] — 8관측 2시드
`spread` +302 %, `spreadear` +34 %, `pairbreak` +139 %(시드별 +227 % / +52 %).
8관측 ctrl s1 기준 원본 .13173 → spread .69244, spreadear .18956, pairbreak .43105.
시드 간 차이가 커서(s4 는 원본 .207 로 나쁜 분지) 5시드가 쌓여야 말할 수 있다.

{ABL_HOWTO}
{LIMITS_ABL}
"""

READMES["20261106_ear_angle_pose"] = """# 귀를 ±90도 각도로 넣는 포즈 파라미터화 (미착수)

창우 2026-11-06 20:01 "귀 방향 yaw는 90도 270도 각각 다른거 아냐?" 에서 나온 설계.
20:37 "그래" 로 승인. **아직 구현하지 않았다** — 이 폴더는 설계와 함정만 담고 있다.

## 아이디어
지금 pose MLP 입력은 `[sin yaw, cos yaw, ear]` 로 귀가 ±1 **라벨**로만 들어간다. 귀 방향이
실제로 헤딩±90도이므로, 그 각도를 포즈 임베딩에도 주면 모델이 라벨이 아니라 방향으로 쓸 수 있다.

## 반드시 피해야 할 함정
라벨을 빼고 각도만 주면 **(헤딩 0, 오른귀)와 (헤딩 180, 왼귀)가 둘 다 90도가 되어 충돌**한다.
두 경우의 HRTF 는 머리 그림자가 좌우 반전이라 실제로 다르다. 그래서 부호는 남기고 각도를
**추가**하는 형태여야 한다: 예 `[sin(yaw + 90e), cos(yaw + 90e), sin yaw, cos yaw, ear]`.

## 하려면
학습이 필요하다(인퍼런스 교란으로는 못 본다). 비용이 가장 싼 Replica 2관측부터:
대조군 `paperrep_ctrl_r2_s0..s4` (.16102, sd .00403), 런당 2.0 h, 5대5.
모델 쪽 플래그 하나 + `core/ckpt.py` 전달 + 기존 모드 비트 동일 스모크가 필요하다
(`../20261106_unshared_encoder/scripts/patch_unsharedenc.py` 가 같은 패턴의 예시다).
"""

READMES["20261106_channel_stacking"] = """# 8채널 스태킹 대조군 (stackall / stackroll)

창우 요청 2026-11-06 18:35 "전부를 하나에 하는 걸로", 18:48 "제대로 실험 돌린건지 검토하고
8 스태킹으로 하는 거 맞는 지 알려줘".

## 질문
리뷰어가 반드시 묻는 "왜 채널을 그냥 쌓지 않고 포즈 붙은 뷰별 토큰으로 가냐"에 **같은 백본으로**
답하는 칸. 기존 답은 백본까지 다른 ResNet(.282)·BatVision(.205) 뿐이었다.

## 두 팔
- `stackall` (커밋 a9d2848) — 8채널을 **한 관측의 8평면**으로. `build()` 가 항등이라 입력
  바이트는 ctrl 과 동일하고 모델만 nviews 1 · in_ch 8 로 세운다. 토큰이 4,096 → 512, 포즈가
  8 → 1(첫 채널의 0도·왼귀), 관측 간 어텐션은 키가 하나라 항등이 된다. 파라미터 15.355658 M
  (ctrl +2,016). 칸 `paperrep_stack_r8_s0..s2`, 168 s/ep.
- `stackroll` (커밋 ce27147) — 공유 인코더를 8번 쓰면서 매번 8채널 전부를 넣고 순서만 돌린다.
  토큰 4,096 · 포즈 8개 · 가중치 공유를 유지한 채 채널이 스템에서 섞인다. ablation 이 아니라
  **새 구조 후보**다. 칸 `paperrep_roll_r8_s0` (1시드 스크리닝).

## 대조군·판정
`paperrep_ctrl_r8_s0..s4` 평균 .14899 (sd .03283). 사전등록 `PREREG_stackall_20261106.md`,
`PREREG_stackroll_20261106.md`. stackall 예측은 ≥.22(50 % 넘게 악화), stackroll 예측은 ctrl ±1 sd.

## 초기 신호 [사실]
같은 시드 ep01 에서 stackall s1 val AbsRel 1.5084 대 ctrl s1 2.8048 로 **스태킹이 앞섰다**.
수렴 속도 차이일 수 있으나 사전등록 예측이 깨질 가능성이 열려 있고, 그렇게 되면 뷰별 토큰화의
근거가 흔들리는 더 중요한 결과다.

## 약한 데
이 ablation 은 뷰별 포즈·레이 임베딩과 뷰 간 어텐션을 **한꺼번에** 뺀다. 둘을 가르려면
`--no-cross` 단독 칸이 필요한데 양 노드 큐에서 한 번도 돌린 적이 없다. 또 subset-aug 의 의미가
관측 마스킹에서 채널 드롭아웃으로 바뀐다(단위와 RNG 스트림은 동일).
"""

READMES["20261106_itd_arm"] = """# ITD 팔 (광대역 단일 지연)

창우 2026-11-06 재차 요청. 위상 전체(IPD) 대신 **광대역 GCC-PHAT 지연 하나**로 접은 입력 평면.

## 구현 (커밋 c21d64b)
`core/pairfeat.py` 에 `itd` 추가. 로더 위상 평면을 단일 지연으로 접어 상수 평면으로 뿌린다
(쿼터 샘플 격자, ±16 샘플, `ITD_MAX`). `in_ch` 2, `needs_ipd` True, dropped 규칙은 `ild` 와 동일.
스모크: 기존 9개 모드 비트 동일, 알려진 지연 −7~+11.75 샘플을 0.25 샘플 안에서 복원.

## 칸
`gibh160_itd_r2_s0` (n1, bs 12 × accum 2 — 24×1 은 ~34 GB 로 3090 에 안 들어간다),
`gibh160_itd_r2_s1..s4` (n14). Gibson 2관측, 런당 11.5 h.

## 사전 예측 [추론]
최적 단일 지연이 위상의 **0.42만 설명**하므로(Gibson 측정) ITD 이득은 IPD(+8.30 %)에 한참 못
미치고 ILD(+3.48 %) 언저리일 것. 리뷰어가 "왜 ITD 를 안 쓰냐"고 물으면 그 0.42 로 답한다.
"""

SYNC = r"""#!/bin/bash
# Fan this node's local artefacts out into the per-experiment results/ dirs in the repo, every 5
# minutes, so each experiment folder is self-contained. Shell loop, no agent. usage: <n1|n14>
NODE=${1:?usage: sync_experiments.sh <n1|n14>}
E=/root/storage/implementation/shared_audio/hear360/experiments
P=/root/local1/changwoo/poseablate_20261106
S=/root/local1/changwoo/relevel_seedfloor_20261016

put() {   # put <src-glob> <exp-dir>
  local d=$E/$2/results/$NODE; mkdir -p "$d"
  cp -f $1 "$d/" 2>/dev/null
}

while :; do
  put "$P/results/poseablate_rep_*_allmetrics_*.json" 20261106_pose_random_heading
  put "$P/results/poseablate_rep_*_yawrot_*.json"     20261106_pose_random_heading
  put "$P/results/poseablate_rep_*_allmetrics_*.json" 20261106_ear_mixing
  put "$P/results/poseablate_rep_*_earmix_*.json"     20261106_ear_mixing
  put "$P/results/poseablate_rep_*_earleft_*.json"    20261106_ear_mixing
  put "$P/results/poseablate_rep_*_pairbreak_*.json"  20261106_channel_directions
  put "$S/logs/paperrep_unshared_*.log"               20261106_unshared_encoder
  put "$S/logs/paperrep_stack_*.log"                  20261106_channel_stacking
  put "$S/logs/paperrep_roll_*.log"                   20261106_channel_stacking
  put "$S/logs/gibh160_itd_*.log"                     20261106_itd_arm
  cp -f "$S"/PREREG_unsharedenc_20261106.md $E/20261106_unshared_encoder/ 2>/dev/null
  cp -f "$S"/PREREG_stack*_20261106.md      $E/20261106_channel_stacking/ 2>/dev/null

  for x in 20261106_unshared_encoder 20261106_pose_random_heading 20261106_ear_mixing \
           20261106_channel_directions 20261106_channel_stacking 20261106_itd_arm; do
    f=$E/$x/results/STATUS_$NODE.txt
    {
      echo "# STATUS $(date '+%Y-%m-%d %H:%M %Z')  node=$NODE  (5분마다 갱신)"
      echo
      echo "## 이 노드에서 도는 우리 학습 칸"
      for g in "$S"/logs/paperrep_unshared_*.log "$S"/logs/paperrep_stack_*.log \
               "$S"/logs/paperrep_roll_*.log "$S"/logs/gibh160_itd_*.log; do
        [ -f "$g" ] || continue
        n=$(grep -c '^\[ep ' "$g"); b=$(basename "$g" .log)
        [ -f "$S/runs/rep/$b/train_done.json" ] || [ -f "$S/runs/mp3d/$b/train_done.json" ] \
          && st=LANDED || st=running
        echo "  $b: $n/40 $st  $(grep '^\[ep ' "$g" | tail -1)"
      done
      echo
      echo "## 교란 채점 패스"
      python - <<'PY' 2>/dev/null
import glob, json, os
for f in sorted(glob.glob("/root/local1/changwoo/poseablate_20261106/results/poseablate_rep_*json")):
    d = json.load(open(f))
    n = sum(len([k for k in v if not k.startswith("_")]) for v in d.values())
    print(f"  {os.path.basename(f)}: cells {len(d)}, mode-evals {n}")
PY
      echo
      echo "## 큐 대기 순서"
      cd "$S" && timeout 150 bash launch_seedfloor.sh --probe 2>/dev/null \
        | grep -E 'READY|SKIP\(running\)' | head -14
    } > "$f" 2>&1
  done
  sleep 300
done
"""

INDEX = """# 2026-11-06 창우 요청 인덱스

각 폴더는 **self-contained** 하다. 질문·설계·칸 이름·대조군·판정 기준·현재 숫자·재현 명령·한계가
그 폴더 README 에 다 있고, 필요한 스크립트는 그 폴더 `scripts/` 에 사본으로 들어 있다.
`results/` 는 양 노드가 5분마다 채우고 git 에는 올라가지 않는다(`results/.gitignore`).
`results/STATUS_n1.txt`·`STATUS_n14.txt` 에 칸별 에폭·패스 진행·큐 대기 순서가 찍힌다.

| 폴더 | 요청 | 종류 | 상태 |
|---|---|---|---|
| `20261106_unshared_encoder` | 19:13 가중치 공유 끊기 + 19:48 Replica 우선 | 학습 | 8관측 s0·s1 진행, 2관측 s0 진행, 나머지 큐 |
| `20261106_pose_random_heading` | 19:48 랜덤헤딩도 Replica | 채점 | 4관측 ctrl 4시드 완료(통과), 회전 스윕 대기 |
| `20261106_ear_mixing` | 20:09 귀가 섞인 채로 inference | 채점 | 4관측 ctrl 4시드 완료, 2관측 진행 |
| `20261106_channel_directions` | 20:20 채널마다 다른 방향, 20:34 짝이 무관한 yaw | 채점 | 8관측 2시드, 무작위 순서판 대기 |
| `20261106_ear_angle_pose` | 20:01 귀를 ±90도 각도로 | 학습 | **미착수** (설계와 함정만) |
| `20261106_channel_stacking` | 18:35 전부 하나로, 18:48 검토 | 학습 | stackall 3칸 거의 착지, stackroll 1칸 진행 |
| `20261106_itd_arm` | ITD 재요청 | 학습 | n1 s0 진행, n14 s1~s4 큐 |

창우가 결정한 것: 용량 맞춘 대조군은 **안 한다**(20:20) — 파라미터 배수는 한계로 명시.
코드 변경은 전부 커밋돼 있고(16a62ba, a9d2848, ce27147, c21d64b) push 는 컨테이너에 깃허브
인증이 없어 막혀 있다.
"""

SCRIPTS = {
    "20261106_unshared_encoder": ["patch_unsharedenc.py", "smoke_unsharedenc.py"],
    "20261106_pose_random_heading": ["pose_ablate2.py", "report_ablate.py", "run_rep_poseablate.sh",
                                     "run_yawrot.sh", "run_repr2_eardep.sh"],
    "20261106_ear_mixing": ["pose_ablate2.py", "report_ablate.py", "run_earmix.sh",
                            "run_repr2_earleft.sh", "run_repr2_eardep.sh"],
    "20261106_channel_directions": ["pose_ablate2.py", "report_ablate.py", "run_pairbreak.sh",
                                    "run_pairbreak_r2_n14.sh"],
    "20261106_ear_angle_pose": [],
    "20261106_channel_stacking": [],
    "20261106_itd_arm": [],
}

GITIGNORE = ("# 결과 json·로그는 NFS 에서 보이게만 두고 git 에는 올리지 않는다 (2026-09-24 규칙)\n"
             "*\n!.gitignore\n")


def main():
    for name, body in READMES.items():
        d = os.path.join(EXP, name)
        os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
        os.makedirs(os.path.join(d, "results"), exist_ok=True)
        open(os.path.join(d, "README.md"), "w").write(body)
        open(os.path.join(d, "results", ".gitignore"), "w").write(GITIGNORE)
        for s in SCRIPTS.get(name, []):
            src = os.path.join(OLD, "scripts", s)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(d, "scripts", s))
            else:
                print(f"  (missing script, skipped) {name}/{s}")
        print(f"ok {name}: README + {len(os.listdir(os.path.join(d, 'scripts')))} scripts")

    open(os.path.join(EXP, "20261106_INDEX.md"), "w").write(INDEX)
    os.makedirs(os.path.join(EXP, "_shared_20261106"), exist_ok=True)
    open(os.path.join(EXP, "_shared_20261106", "sync_experiments.sh"), "w").write(SYNC)
    os.chmod(os.path.join(EXP, "_shared_20261106", "sync_experiments.sh"), 0o755)
    for s in ("PREREG_unsharedenc_20261106.md",):
        src = os.path.join(OLD, s)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(EXP, "20261106_unshared_encoder", s))
    print("index + sync written")


if __name__ == "__main__":
    main()
