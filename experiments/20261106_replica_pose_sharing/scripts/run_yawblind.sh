#!/bin/bash
# 헤딩만 제거(yawblind): 모든 yaw 를 0 으로, 귀 부호는 원본 유지.
cd /root/storage/implementation/shared_audio/hear360
PY=/opt/conda/envs/shared_audio/bin/python
G=/root/storage/implementation/shared_audio/hear360/comparison_gibson
O=/root/local1/changwoo/poseablate_20261106/results
LOG=/root/local1/changwoo/poseablate_20261106/run_yawblind.log
mkdir -p "$O"
export CUDA_VISIBLE_DEVICES=0 EVAL_BS=4 ABL_MODES=yawblind
{
echo "[$(date -u +%FT%TZ)] 헤딩만 제거 4관측 시작"
"$PY" /tmp/pose_ablate2.py gibson \
  "$G/_from_n1/gibh160_ctrl_fb_s3,$G/_from_n1/gibh160_pairild_fb_s1" \
  "$O/poseablate_gib_fb_allmetrics_20261106.json"
echo "[$(date -u +%FT%TZ)] 헤딩만 제거 8관측 시작"
"$PY" /tmp/pose_ablate2.py gibson \
  "$G/_from_n1/gibh160_ctrl_r8_s0,$G/_from_n1/gibh160_pairild_r8_s0" \
  "$O/poseablate_gib_r8_allmetrics_20261106.json"
echo "[$(date -u +%FT%TZ)] 끝"
touch "$O/DONE_YAWBLIND"
} 2>&1 | tee -a "$LOG"
exec bash
