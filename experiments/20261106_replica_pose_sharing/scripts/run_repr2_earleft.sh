#!/bin/bash
# Replica 2 obs, the one ear configuration no other pass covers: BOTH ears left.
# At 2 observations randear and earflipk all collapse onto `swap`, which the 4-mode pass measured,
# so the only new information here is the constant-left assignment (earblind is constant-right).
cd /root/storage/implementation/shared_audio/hear360 || exit 1
P=/root/local1/changwoo/poseablate_20261106
E=/root/local1/changwoo/relevel_seedfloor_20261016/runs/rep
export CUDA_VISIBLE_DEVICES=6 EVAL_BS=4 ABL_MODES=true,earleft

pick() { local out=""; for d in "$@"; do [ -f "$d/best.pth" ] && out="$out,$d"; done; echo "${out#,}"; }
R2=$(pick "$E"/paperrep_ctrl_r2_s* "$E"/paperrep_ild_r2_s* "$E"/paperrep_ipd_r2_s* \
          "$E"/paperrep_ildipd_r2_s* "$E"/paperrep_ildmin_r2_s*)

echo "[$(date -u +%FT%TZ)] Replica 2 obs, constant-left ears, 5 arms x 5 seeds"
/opt/conda/envs/shared_audio/bin/python "$P/pose_ablate2.py" replica "$R2" \
  "$P/results/poseablate_rep_r2_earleft_20261106_n14.json"
echo "[$(date -u +%FT%TZ)] done"
touch "$P/results/DONE_REP_R2_EARLEFT"
