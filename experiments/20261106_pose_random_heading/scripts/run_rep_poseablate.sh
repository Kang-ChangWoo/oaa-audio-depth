#!/bin/bash
# Replica pose ablation (inference only, training 0), 2026-11-06.
# Changwoo 19:48 KST: the random-heading ablation belongs on Replica too, not only Gibson.
# Script is the yawblind-capable pose_ablate2.py; it resumes, so a kill only costs the current mode.
# NOTE: n1's /root/local1 is a DIFFERENT disk from n14's, so this json is the n1 copy by name.
cd /root/storage/implementation/shared_audio/hear360 || exit 1
PY=/opt/conda/envs/shared_audio/bin/python
E=/root/local1/changwoo/relevel_seedfloor_20261016/runs/rep
O=/root/local1/changwoo/poseablate_20261106/results
mkdir -p "$O"
cp -n /tmp/pose_ablate2.py /root/local1/changwoo/poseablate_20261106/pose_ablate2.py 2>/dev/null
S=/root/local1/changwoo/poseablate_20261106/pose_ablate2.py

export CUDA_VISIBLE_DEVICES=1 EVAL_BS=4
export ABL_MODES=true,randyaw,randyaw2,randyaw3,yawblind,yawroll,swap,blind,earblind

pick() {                     # only run dirs that actually have a best.pth
  local out=""
  for d in "$@"; do [ -f "$d/best.pth" ] && out="$out,$d"; done
  echo "${out#,}"
}

FB=$(pick "$E"/paperrep_ctrl_fb_s* "$E"/paperrep_ild_fb_s*)
R8=$(pick "$E"/paperrep_ctrl_r8_s* "$E"/paperrep_ild_r8_s*)

echo "[$(date -u +%FT%TZ)] Replica 4 obs: $FB"
"$PY" "$S" replica "$FB" "$O/poseablate_rep_fb_allmetrics_20261106_n1.json"
echo "[$(date -u +%FT%TZ)] Replica 8 obs: $R8"
"$PY" "$S" replica "$R8" "$O/poseablate_rep_r8_allmetrics_20261106_n1.json"
echo "[$(date -u +%FT%TZ)] done"
touch "$O/DONE_REP_POSEABLATE"
