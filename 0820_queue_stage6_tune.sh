#!/bin/bash
# 0820 Stage-6: the hyperparameters 116 runs never touched, plus the overfitting the val curves show.
#
# Evidence this queue is built on:
#
#  (a) EVERY MP3D run peaks between epoch 12 and 16 of 40 and then degrades monotonically:
#      h44_cnn_fb   0.8831 @ep16 -> 0.9925 @ep39   (+12.4%)
#      h44_sslnat   0.8508 @ep12 -> 0.9139 @ep25   (+7.4%)
#      h44_sslcs    0.8685 @ep14 -> 0.9078 @ep26   (+4.5%)
#      C3_r8        0.8814 @ep15 -> 0.9299 @ep24   (+5.5%)
#      The best checkpoint is kept, so reported numbers are valid -- but two thirds of the budget is
#      spent past the optimum, the cosine schedule is scaled to an epoch count the run never uses,
#      and weight decay has never been tuned. The CNN overfits hardest, which is consistent with
#      pretraining acting as a regulariser.
#
#  (b) A survey of the args of all 116 campaign runs found four knobs never swept even once:
#      afm_lr_ratio  0.1 in 108/108 AFM runs   <- the central fine-tuning knob for a pretrained
#                                                 backbone; with LLRD 0.75 on top, block 0 sits at
#                                                 2.1e-6, i.e. effectively frozen
#      wd            1e-4 in 108/108
#      rounds        2 in 108/108              <- fusion capacity, and fusion is where the measured
#                                                 bottleneck is (MP3D 8ch, see 0820_RESULTS.md)
#      dim           256 in 108/108
#      loss_absrel   0.0 in 61, two one-off runs -- near-field is the band the CNN still owns
#
# Baselines to beat (same recipe, hop 160): sslam+LLRD+cs  MP3D fb 0.7744 | Replica fb 0.2573+-.002
#
#   bash 0820_queue_stage6_tune.sh
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv"
FBMP="--nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"
FBREP="--nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"
R8MP="--nviews 8 --data-mode r8 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"

JOBS=(
# --- (a) the overfitting the curves show: regularisation and a schedule matched to where runs peak
"0820_tune_wd1e3_fb_mp3d|data_mp3d|$SL $FBMP --wd 1e-3"
"0820_tune_wd1e2_fb_mp3d|data_mp3d|$SL $FBMP --wd 1e-2"
"0820_tune_ep20_fb_mp3d|data_mp3d|$SL --nviews 4 --data-mode fb --epochs 20 --batch-size 8 --accum 4 --warmup-ep 4"
# --- (b) afm_lr_ratio: how much the pretrained backbone is allowed to move. Never swept, either way.
"0820_tune_lrr02_fb_mp3d|data_mp3d|$SL $FBMP --afm-lr-ratio 0.2"
"0820_tune_lrr02_fb_rep|data_0422|$SL $FBREP --afm-lr-ratio 0.2"
"0820_tune_lrr005_fb_rep|data_0422|$SL $FBREP --afm-lr-ratio 0.05"
# --- (b) fusion capacity, on the cell where fusion is the measured bottleneck
"0820_tune_rounds3_r8_mp3d|data_mp3d|$SL $R8MP --rounds 3"
# --- (b) near-field is the band the CNN still owns; AbsRel upweights it
"0820_tune_absrel_fb_mp3d|data_mp3d|$SL $FBMP --loss-absrel 0.1"
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
echo "[dispatch] all ${#JOBS[@]} stage-6 tuning jobs launched"
