#!/bin/bash
# Replica 2-observation ear-dependence (inference only, training 0), 2026-11-06.
# The Gibson 2-obs verdict (5 arms x 5 seeds) now gets its Replica twin, which is the dataset
# Changwoo cares about. At 2 observations both channels carry yaw 0 and differ only by the ear
# sign, so the yaw-permutation modes are the identity; the ear direction (heading +-90 deg) lives
# in the ear sign and in the geometry bias column r.(R(yaw)x_hat)*ear. `swap` is therefore THE
# ear-direction perturbation here, and `earblind` collapses onto `blind`.
# n14's /root/local1 is a different disk from n1's, hence the _n14 suffix on the json.
cd /root/storage/implementation/shared_audio/hear360 || exit 1
PY=/opt/conda/envs/shared_audio/bin/python
E=/root/local1/changwoo/relevel_seedfloor_20261016/runs/rep
O=/root/local1/changwoo/poseablate_20261106/results
S=/root/local1/changwoo/poseablate_20261106/pose_ablate2.py
mkdir -p "$O"

export CUDA_VISIBLE_DEVICES=1 EVAL_BS=4
export ABL_MODES=true,swap,blind,earblind

pick() { local out=""; for d in "$@"; do [ -f "$d/best.pth" ] && out="$out,$d"; done; echo "${out#,}"; }
R2=$(pick "$E"/paperrep_ctrl_r2_s* "$E"/paperrep_ild_r2_s* "$E"/paperrep_ipd_r2_s* \
          "$E"/paperrep_ildipd_r2_s* "$E"/paperrep_ildmin_r2_s*)

echo "[$(date -u +%FT%TZ)] Replica 2 obs, 5 arms x 5 seeds"
"$PY" "$S" replica "$R2" "$O/poseablate_rep_r2_allmetrics_20261106_n14.json"
echo "[$(date -u +%FT%TZ)] done"
touch "$O/DONE_REP_R2_EARDEP"
