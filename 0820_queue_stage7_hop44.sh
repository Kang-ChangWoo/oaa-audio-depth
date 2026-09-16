#!/bin/bash
# 0820 Stage-7: carry the input-resolution fix to the remaining six cells.
#
# Established so far, on Replica 4ch only:
#   hop 160 -> 44 (18 -> 64 real STFT frames, N_FFT and window unchanged, zero added parameters)
#     OAA-CNN        0.2596 -> 0.2496   (-0.0100)
#     sslam+LLRD+cs  0.2573 -> 0.2496   (-0.0077)
#     sslam+native   0.2685 -> 0.2474   (-0.0211)
#   The CNN gains as much as the AFM, so this is an input-level shift, not an encoder result.
#
# One cell is not a result. This queue runs the unified configuration (sslam + LLRD + conv stem) at
# hop 44 across the six cells that have no hop-44 number yet, with a CNN control on the two that
# decide the most. Ordered so the decisive cells finish first.
#
# MP3D 8ch leads because it is the campaign's only rule-level loss AND the cell whose failure we
# diagnosed: the encoder emits mic-specific structure (dratio 0.43) that is lost when eight
# observations are combined. There is a mechanism linking that to the input -- at hop 160 the
# distance axis is 28x replicated, so the per-mic differences D_i = S_i - mean_j S_j are small to
# begin with. Denser time sampling should make them larger and harder to collapse. That is a
# PREDICTION, and analysis/diff_stats.py tests it directly: if hop 44 raises dratio on this cell,
# the input and the fusion failure are the same problem; if dratio is unchanged while MAE moves (or
# vice versa), they are separate and must be fixed separately.
#
#   bash 0820_queue_stage7_hop44.sh
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv"
CN="--audio-backbone cnn --warmup-ep 4"
R2="--nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2"
R6REP="--nviews 6 --data-mode r6 --epochs 40 --batch-size 4 --accum 8"
R8REP="--nviews 8 --data-mode r8 --epochs 40 --batch-size 3 --accum 11 --subset-aug --vdrop-kmax 4"
R6MP="--nviews 6 --data-mode r6 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"
R8MP="--nviews 8 --data-mode r8 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"

JOBS=(
# --- the loss cell first, with its CNN control
"0820_h44_sslcs_r8_mp3d|data_mp3d|$SL $R8MP"
"0820_h44_cnn_r8_mp3d|data_mp3d|$CN $R8MP"
# --- the rest of MP3D
"0820_h44_sslcs_r6_mp3d|data_mp3d|$SL $R6MP"
"0820_h44_sslcs_r2_mp3d|data_mp3d|$SL $R2"
# --- Replica, with a control on r8 (the cell where the released recipe is strongest)
"0820_h44_sslcs_r8_rep|data_0422|$SL $R8REP"
"0820_h44_cnn_r8_rep|data_0422|$CN $R8REP"
"0820_h44_sslcs_r6_rep|data_0422|$SL $R6REP"
"0820_h44_sslcs_r2_rep|data_0422|$SL $R2"
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
      CUDA_VISIBLE_DEVICES=$g DATA_MODULE=$dm STFT_HOP=44 setsid nohup python3 0820_train_oaa_afm.py \
        --run-name $name $extra --lr 5e-4 --out-dir comparison_0820 \
        > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      i=$((i+1)); sleep 90
    fi
  done
  sleep 180
done
echo "[dispatch] all ${#JOBS[@]} stage-7 hop-44 jobs launched"
