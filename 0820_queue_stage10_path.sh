#!/bin/bash
# 0820 Stage-10: where does the hop-44 gain actually land — the pretrained AFM, or the fine CNN?
#
# THE PUZZLE. Three measurements do not fit the story the report currently tells.
#
#   1. hop 160 already satisfies hop <= win/2 = 200, so hop 44 is ~4.5x oversampling and adds
#      almost no genuinely new information.
#   2. Re-allocating the AFM's 512 tokens to match the real frame count (32 freq x 16 time instead
#      of 16 x 32, identical parameter count, data untouched) recovers only -0.0008 of the -0.0077
#      that hop 44 is worth on Replica 4ch. So it is not token waste on the AFM side either.
#   3. The PURE CNN gains MORE from hop 44 than the AFM model does (-0.0100 vs -0.0077).
#
# (3) is the clue. The AFM model is not AFM-only: it carries the original fine CNN branch, which
# also consumes the 256x512 spectrogram. If the gain belongs to CNN-style processing, then inside
# the AFM model it is the FINE branch collecting it, not the pretrained transformer — and the
# report's "the input matters more than the encoder" line needs the qualifier "and the input gain
# goes to the CNN path, not the foundation model".
#
# THE EXPERIMENT. --fine-frames / --afm-frames coarsen ONE branch's time axis to n distinct values
# and restore the width, emulating a coarser hop for that branch alone (verified: a hop-44 input at
# n=18 leaves exactly 18 distinct columns, matching the true hop-160 frame count). Parameters are
# unchanged in every arm. Run at hop 44, so:
#
#   arm            fine branch   AFM branch   reads as
#   (a) existing   hop 44        hop 44       both fed the fine input   -> Rep 0.2496 / MP3D 0.7562
#   (b) fine18     hop 160       hop 44       only the AFM gets it
#   (c) afm18      hop 44        hop 160      only the fine CNN gets it
#   (d) existing   hop 160       hop 160      neither                   -> Rep 0.2573 / MP3D 0.7744
#
# If (c) ~ (a) and (b) ~ (d), the gain lives in the fine CNN. If reversed, in the AFM. If both arms
# land mid-way, the two branches share it and neither story is clean.
#
# Cells: Replica 4ch and MP3D 4ch — both have (a) and (d) already measured, so each new run has two
# anchors and only two runs per cell are needed.
#
#   bash 0820_queue_stage10_path.sh
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv"
FB="--nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"

JOBS=(
"0820_s10_fine18_rep|data_0422|$SL $FB --fine-frames 18"
"0820_s10_afm18_rep|data_0422|$SL $FB --afm-frames 18"
"0820_s10_fine18_mp3d|data_mp3d|$SL $FB --fine-frames 18"
"0820_s10_afm18_mp3d|data_mp3d|$SL $FB --afm-frames 18"
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
      CUDA_VISIBLE_DEVICES=$g DATA_MODULE=$dm STFT_HOP=44 STFT_WIN=400 setsid nohup \
        python3 0820_train_oaa_afm.py --run-name $name $extra --lr 5e-4 --out-dir comparison_0820 \
        > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      i=$((i+1)); sleep 90
    fi
  done
  sleep 180
done
echo "[dispatch] all ${#JOBS[@]} stage-10 path jobs launched"
