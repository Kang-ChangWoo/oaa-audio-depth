#!/bin/bash
# 0820 Stage-1 screening: 5 AFM backbones x {Replica fb, MP3D fb} (4 observations).
# GPUs 2-6: one MP3D run each (long pole); GPU 7: the five Replica runs sequentially.
# CNN reference = existing comparison/oaa_fb_fin (Replica) and comparison_mp3d/oaa_fb_fin (MP3D).
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
BB=(audiomosaic bat eat sslam m2d)
G=(2 3 4 5 6)
for i in "${!BB[@]}"; do b=${BB[$i]}; g=${G[$i]}
  CUDA_VISIBLE_DEVICES=$g DATA_MODULE=data_mp3d setsid nohup python3 0820_train_oaa_afm.py \
    --run-name 0820_${b}_fb_mp3d --audio-backbone $b --nviews 4 --data-mode fb \
    --lr 5e-4 --warmup-ep 4 --epochs 40 --batch-size 8 --accum 1 --seed 0 \
    --out-dir comparison_0820 > comparison_0820/logs/0820_${b}_fb_mp3d.log 2>&1 < /dev/null &
done
CUDA_VISIBLE_DEVICES=7 DATA_MODULE=data_0422 setsid nohup bash -c '
  for b in audiomosaic bat eat sslam m2d; do
    python3 0820_train_oaa_afm.py --run-name 0820_${b}_fb_rep --audio-backbone $b --nviews 4 --data-mode fb \
      --lr 5e-4 --warmup-ep 4 --epochs 40 --batch-size 8 --accum 4 --seed 0 \
      --out-dir comparison_0820 > comparison_0820/logs/0820_${b}_fb_rep.log 2>&1
  done' < /dev/null > /dev/null 2>&1 &
echo "queued: 5 MP3D runs on GPUs 2-6, 5 Replica runs chained on GPU 7"
