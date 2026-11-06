#!/bin/bash
# Replica ear-direction MIXING at inference (training 0), 2026-11-06.
# Changwoo 20:09 KST: test the case where the two ear directions are not the correct 90/270 pair
# but mixed up. `randear*` permutes the ear column across channels with the identity rejected,
# `earflipk` flips the last k headings' ears (dose-response), `earleft` pins every channel to the
# left ear - the constant that `earblind` (all right) leaves untested.
# Mode lists differ per observation count because at 2 headings earflip2 IS swap, which the
# 9-mode pass already measured. Separate json from that pass: two processes, two files.
cd /root/storage/implementation/shared_audio/hear360 || exit 1
PY=/opt/conda/envs/shared_audio/bin/python
E=/root/local1/changwoo/relevel_seedfloor_20261016/runs/rep
O=/root/local1/changwoo/poseablate_20261106/results
S=/root/local1/changwoo/poseablate_20261106/pose_ablate2.py

export CUDA_VISIBLE_DEVICES=5 EVAL_BS=4

pick() { local out=""; for d in "$@"; do [ -f "$d/best.pth" ] && out="$out,$d"; done; echo "${out#,}"; }
FB=$(pick "$E"/paperrep_ctrl_fb_s* "$E"/paperrep_ild_fb_s*)
R8=$(pick "$E"/paperrep_ctrl_r8_s* "$E"/paperrep_ild_r8_s*)

echo "[$(date -u +%FT%TZ)] ear-mix 4 obs (2 headings)"
ABL_MODES=true,randear,randear2,randear3,earflip1,earleft \
  "$PY" "$S" replica "$FB" "$O/poseablate_rep_fb_earmix_20261106_n1.json"
echo "[$(date -u +%FT%TZ)] ear-mix 8 obs (4 headings)"
ABL_MODES=true,randear,randear2,randear3,earflip1,earflip2,earflip3,earleft \
  "$PY" "$S" replica "$R8" "$O/poseablate_rep_r8_earmix_20261106_n1.json"
echo "[$(date -u +%FT%TZ)] done"
touch "$O/DONE_REP_EARMIX"
