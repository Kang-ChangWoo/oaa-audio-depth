#!/bin/bash
# 0820 Stage-3r: 20 runs closing the coverage gaps found on 2026-09-11.
#
# Why these twenty (in dispatch order):
#
#  B. novd twins for the mic-drop degradation study (2 runs, first because they unblock
#     analysis/micdrop_study/). The unified setting and the convstem variant only exist at
#     Replica r8 WITH vdrop, so neither can enter the like-for-like (no-mic-drop) comparison
#     against prior work. These give them a no-vdrop twin.
#
#  A. sslam + LLRD + convstem, all 8 cells (8 runs). The campaign's single largest blind spot:
#     convstem is the strongest add-on (3 strict wins, 2 cell records) but was only ever bolted
#     onto eat; sslam is the strongest Replica backbone. Their intersection is untested. The one
#     sslam+cs run that exists (Rep fb, early screening, 0.3017) predates LLRD, which is the
#     one intervention known to be unconditionally required for post-norm backbones — so it
#     does not settle the combination.
#
#  C. Seed-hardening of every strict-win cell that currently rests on a single seed (6 runs).
#     This campaign has already retracted two single-seed wins (Replica fb 0.2553 -> 3-seed tie,
#     llrd65 MP3D fb 0.7714 -> 2-seed tie). Six of the surviving headline wins are still n=1.
#
#  D. LLRD decay probe (2 runs). Only 0.75 (default) and 0.65 (MP3D fb) were tried. On Replica
#     r2/r6 LLRD costs 0.005-0.009; a gentler 0.85 tests whether that cost is decay-rate driven.
#
#  E. Two seed replicates that decide interpretations rather than cells (2 runs): the only
#     convstem failure (Rep fb 0.2644) and the headline cell of the new A combination.
#
# Recipes mirror the proven per-cell settings exactly (see the matching runs in comparison_0820):
#   r2  ep40 bs12 accum2 | fb  ep40 bs8 accum4
#   r6  Rep ep40 bs4 accum8   | r6/r8 MP3D ep30 bs4 accum1 --stem-stride1
#   r8  Rep ep40 bs3 accum11  | vdrop only at Replica r8 (the established vdrop law)
# LLRD runs use warmup 8; plain-sslam runs use warmup 4.
#
#   bash 0820_queue_stage3r.sh            # empty GPUs (<2000MB) only, one job per free slot
#   GPUS="0 1" bash 0820_queue_stage3r.sh
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8"
EL="--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8"
S0="--audio-backbone sslam --warmup-ep 4"

JOBS=(
# --- B: no-vdrop twins that complete the degradation-study roster
"0820_sslamllrd_r8novd_rep|data_0422|$SL --nviews 8 --data-mode r8 --epochs 40 --batch-size 3 --accum 11"
"0820_eatllrd_cs_r8novd_rep|data_0422|$EL --afm-stem conv --nviews 8 --data-mode r8 --epochs 40 --batch-size 3 --accum 11"

# --- A: sslam + LLRD + convstem across all eight cells
"0820_sslamllrd_cs_r2_rep|data_0422|$SL --afm-stem conv --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2"
"0820_sslamllrd_cs_fb_rep|data_0422|$SL --afm-stem conv --nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"
"0820_sslamllrd_cs_r6_rep|data_0422|$SL --afm-stem conv --nviews 6 --data-mode r6 --epochs 40 --batch-size 4 --accum 8"
"0820_sslamllrd_cs_r8_rep|data_0422|$SL --afm-stem conv --nviews 8 --data-mode r8 --epochs 40 --batch-size 3 --accum 11 --subset-aug --vdrop-kmax 4"
"0820_sslamllrd_cs_r2_mp3d|data_mp3d|$SL --afm-stem conv --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2"
"0820_sslamllrd_cs_fb_mp3d|data_mp3d|$SL --afm-stem conv --nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"
"0820_sslamllrd_cs_r6_mp3d|data_mp3d|$SL --afm-stem conv --nviews 6 --data-mode r6 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"
"0820_sslamllrd_cs_r8_mp3d|data_mp3d|$SL --afm-stem conv --nviews 8 --data-mode r8 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"

# --- C: seed 1 for the six strict wins that are still n=1
"0820_sslam_r2_rep_s1|data_0422|$S0 --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2 --seed 1"
"0820_sslam_r2_mp3d_s1|data_mp3d|$S0 --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2 --seed 1"
"0820_eatllrd_r2_rep_s1|data_0422|$EL --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2 --seed 1"
"0820_eatllrd_cs_r2_rep_s1|data_0422|$EL --afm-stem conv --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2 --seed 1"
"0820_eatllrd_cs_r2_mp3d_s1|data_mp3d|$EL --afm-stem conv --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2 --seed 1"
"0820_eatllrd_cs_fb_mp3d_s1|data_mp3d|$EL --afm-stem conv --nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4 --seed 1"

# --- D: does a gentler layer-decay remove LLRD's Replica cost?
"0820_sslam_llrd85_r2_rep|data_0422|--audio-backbone sslam --afm-llrd 0.85 --warmup-ep 8 --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2"
"0820_sslam_llrd85_r6_rep|data_0422|--audio-backbone sslam --afm-llrd 0.85 --warmup-ep 8 --nviews 6 --data-mode r6 --epochs 40 --batch-size 4 --accum 8"

# --- E: seed replicates that settle interpretations
"0820_eatllrd_cs_fb_rep_s1|data_0422|$EL --afm-stem conv --nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4 --seed 1"
"0820_sslamllrd_cs_fb_rep_s1|data_0422|$SL --afm-stem conv --nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4 --seed 1"
)

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for g in $GPUS; do
    [ $i -ge ${#JOBS[@]} ] && break
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$mem" -lt 2000 ]; then
      IFS='|' read -r name dm extra <<< "${JOBS[$i]}"
      if [ -e "comparison_0820/logs/$name.log" ]; then i=$((i+1)); continue; fi
      echo "[dispatch] $name -> GPU $g ($(date +%m/%d\ %H:%M))"
      CUDA_VISIBLE_DEVICES=$g DATA_MODULE=$dm setsid nohup python3 0820_train_oaa_afm.py \
        --run-name $name $extra --lr 5e-4 --out-dir comparison_0820 \
        > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      i=$((i+1)); sleep 90
    fi
  done
  sleep 180
done
echo "[dispatch] all ${#JOBS[@]} stage-3r jobs launched"
