#!/bin/bash
# Replica: break the (heading, L/R) ear structure at inference (training 0), 2026-11-06.
# Changwoo 20:20 KST: give each channel a different direction so the ear form is not preserved.
#   spread     n evenly spaced yaws, constant ear sign  -> no pair survives, no shared heading
#   spreadear  the same yaws with the original ear signs -> pairing gone, ear label kept
#   pairbreak  yaw column rolled by one channel         -> pairing gone, yaw values stay INSIDE
#              the trained set, so this one is free of the unseen-angle confound
# At 8 channels `spread` uses 45 deg steps, which training never saw; at 4 channels it uses
# 90/270, which the fb rig never saw either. `pairbreak` is the control for exactly that.
cd /root/storage/implementation/shared_audio/hear360 || exit 1
PY=/opt/conda/envs/shared_audio/bin/python
E=/root/local1/changwoo/relevel_seedfloor_20261016/runs/rep
O=/root/local1/changwoo/poseablate_20261106/results
S=/root/local1/changwoo/poseablate_20261106/pose_ablate2.py

export CUDA_VISIBLE_DEVICES=5 EVAL_BS=4 ABL_MODES=true,yawrot45,yawrot90,yawrot135,yawrot180

pick() { local out=""; for d in "$@"; do [ -f "$d/best.pth" ] && out="$out,$d"; done; echo "${out#,}"; }
FB=$(pick "$E"/paperrep_ctrl_fb_s* "$E"/paperrep_ild_fb_s*)
R8=$(pick "$E"/paperrep_ctrl_r8_s* "$E"/paperrep_ild_r8_s*)

echo "[$(date -u +%FT%TZ)] pair-break 8 obs first (4 headings, the case Changwoo asked about)"
"$PY" "$S" replica "$R8" "$O/poseablate_rep_r8_yawrot_20261106_n1.json"
echo "[$(date -u +%FT%TZ)] pair-break 4 obs"
"$PY" "$S" replica "$FB" "$O/poseablate_rep_fb_yawrot_20261106_n1.json"
echo "[$(date -u +%FT%TZ)] done"
touch "$O/DONE_REP_YAWROT"
