# 귀 방향을 거짓으로 주는 교란 (Replica, 학습 0)

창우 요청 2026-11-06 20:09: "양쪽 귀가 90 270 도가 아니라 섞여서 inference 하는 것도 학습없이
테스트해보긴 해야될 거 같아".

## 배경 — 귀 방향은 어디에 들어 있나
포즈는 `[sin yaw, cos yaw, ear]` 이고 yaw 는 **머리 헤딩**, 귀는 ±1 라벨이다. 귀 방향 자체는
기하 바이어스의 두 번째 열 `r·(R(yaw)x̂)·ear` 로 들어간다(`model/oaa.py` 757~772행). 즉 귀 축은
헤딩을 90도 돌린 축이고, 왼귀는 헤딩−90도 · 오른귀는 +90도를 본다.

## 모드
`swap` 귀 부호 전부 뒤집기(90↔270) / `earblind` 전부 오른귀 / `earleft` 전부 왼귀 /
`randear`·`randear2`·`randear3` 귀 부호를 채널 간 순열(항등 배열은 거부) /
`earflip{k}` 뒤쪽 헤딩 k 개만 뒤집기 / `blind` 포즈 전부 동일.

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

