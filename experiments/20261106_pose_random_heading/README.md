# 랜덤 헤딩 포즈 교란 (Replica, 학습 0)

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

