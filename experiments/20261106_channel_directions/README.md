# 채널마다 다른 방향 — 귀 형태를 깨는 교란 (Replica, 학습 0)

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

## 재현 (학습 0, 채점만)
```
cd /root/storage/implementation/shared_audio/hear360
export CUDA_VISIBLE_DEVICES=<카드> EVAL_BS=4 ABL_MODES=<모드,모드,...>
/opt/conda/envs/shared_audio/bin/python scripts/pose_ablate2.py replica \
  <run_dir>,<run_dir>,... results/<out>.json
```
`pose_ablate2.py` 는 체크포인트에서 포즈를 복원하고 채점할 때 리스트만 바꿔 끼운다. 같은 json 에
이어쓰므로 중간에 끊겨도 손해가 없고, 이미 있는 모드는 건너뛴다. 지표 11종 전부 저장한다.

**검산**: `true` 모드는 그 체크포인트의 기록된 test AbsRel 과 일치해야 한다. 지금까지 3/3 일치
(`paperrep_ctrl_fb_s1` .14360, `ctrl_fb_s2` .14962, `ctrl_r8_s4` .20713).

**2관측에서 무효인 모드**: yaw 가 둘 다 0 이므로 `randyaw*`·`pairbreak` 는 항등이고
`randear*`·`earflip*` 는 `swap` 과 같아진다. 스크립트가 `_identity_noop` 으로 표시한다.

## 한계
- 각 칸 1시드가 아니라 가능한 시드를 전부 돌리지만, 8관측은 체크포인트가 ctrl·ILD 각 2시드뿐이다.
- 포즈 교란은 +200 % 이상 포화 구간이라 팔 간(기본 입력 대 ILD) 차이는 읽으면 안 된다.
- 비율 지표(AbsRel)가 가장 크게 움직이고 RMSE 는 완만하다. 밴드로 쪼개 해석하지 말 것
  (주지표는 전체 AbsRel).

