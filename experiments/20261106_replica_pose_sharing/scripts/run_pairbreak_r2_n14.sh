#!/bin/bash
# Replica: break the (heading, L/R) ear structure at inference (training 0), 2026-11-06.
# Changwoo 20:20 KST: give each channel a different direction so the ear form is not preserved.
#   spread     n evenly spaced yaws, constant ear sign  -> no pair survives, no shared heading
#   spreadear  the same yaws with the original ear signs -> pairing gone, ear label kept
#   pairbreak  yaw column rolled by one channel         -> pairing gone, yaw values stay INSIDE
#              the trained set, so this one is free of the unseen-angle confound
# At 2 channels `spread` gives (0,R),(180,R): the pairing is gone and 180 deg was never seen
# in training either, so this is the unseen-angle case by construction. `pairbreak` is a no-op
# at 2 channels (one heading) and is left out.
# At 8 channels `spread` uses 45 deg steps, which training never saw; at 4 channels it uses
# 90/270, which the fb rig never saw either. `pairbreak` is the control for exactly that.
cd /root/storage/implementation/shared_audio/hear360 || exit 1
PY=/opt/conda/envs/shared_audio/bin/python
E=/root/local1/changwoo/relevel_seedfloor_20261016/runs/rep
O=/root/local1/changwoo/poseablate_20261106/results
S=/root/local1/changwoo/poseablate_20261106/pose_ablate2.py

export CUDA_VISIBLE_DEVICES=7 EVAL_BS=4 ABL_MODES=${ABL_MODES:-true,spread,spreadear}

pick() { local out=""; for d in "$@"; do [ -f "$d/best.pth" ] && out="$out,$d"; done; echo "${out#,}"; }
R2=$(pick "$E"/paperrep_ctrl_r2_s* "$E"/paperrep_ild_r2_s* "$E"/paperrep_ipd_r2_s* \
          "$E"/paperrep_ildipd_r2_s* "$E"/paperrep_ildmin_r2_s*)

echo "[$(date -u +%FT%TZ)] Replica 2 obs, pair structure broken, 5 arms x 5 seeds"
"$PY" "$S" replica "$R2" "$O/poseablate_rep_r2_pairbreak_20261106_n14.json"
echo "[$(date -u +%FT%TZ)] done"
touch "$O/DONE_REP_R2_PAIRBREAK"
