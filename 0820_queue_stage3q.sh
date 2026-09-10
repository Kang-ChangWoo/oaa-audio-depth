#!/bin/bash
# 0820 Stage-3q: rerun of sslam_llrd65_fb_mp3d_s1 (original dispatch died at startup with CUDA OOM
# after a co-tenant grabbed the GPU). Arbiter seed for the llrd65 MP3D fb verdict
# (s0 0.7714 / s2 0.7892 -> 2-seed mean 0.7803 = tie; s1 decides). Empty-GPU (<2000MB) only.
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew
mkdir -p comparison_0820/logs
name=0820_sslam_llrd65_fb_mp3d_s1
while :; do
  for g in 0 1 2 3 4 5 6 7; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$mem" -lt 2000 ] && [ ! -e "comparison_0820/logs/$name.log" ]; then
      echo "[dispatch] $name -> GPU $g ($(date +%m/%d\ %H:%M))"
      CUDA_VISIBLE_DEVICES=$g DATA_MODULE=data_mp3d setsid nohup python3 0820_train_oaa_afm.py \
        --run-name $name --audio-backbone sslam --afm-llrd 0.65 --warmup-ep 8 \
        --nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4 --seed 1 \
        --lr 5e-4 --out-dir comparison_0820 > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      exit 0
    fi
  done
  [ -e "comparison_0820/logs/$name.log" ] && exit 0
  sleep 180
done
