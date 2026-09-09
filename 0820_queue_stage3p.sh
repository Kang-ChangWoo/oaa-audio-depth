#!/bin/bash
# 0820 Stage-3p: retry of the failed unified-setting MP3D r8 cell (sslam_llrd_r8_mp3d
# collapsed to a ~1.11 val plateau at seed 0; test 0.9861). Seed 1, same recipe.
# Waits until ALL stage-3n beyond jobs are dispatched (user priority), then empty-GPU only.
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew
mkdir -p comparison_0820/logs
BEYOND="0820_beyond_fb_rep 0820_beyond_fb_mp3d 0820_beyond_r2_rep 0820_beyond_r2_mp3d 0820_beyond_r6_rep 0820_beyond_r6_mp3d 0820_beyond_r8_rep 0820_beyond_r8_mp3d"
while :; do
  ok=1; for b in $BEYOND; do [ -e "comparison_0820/logs/$b.log" ] || { ok=0; break; }; done
  [ $ok -eq 1 ] && break; sleep 300
done
name=0820_sslam_llrd_r8_mp3d_s1
while :; do
  for g in 0 1 2 3 4 5 6 7; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$mem" -lt 2000 ] && [ ! -e "comparison_0820/logs/$name.log" ]; then
      echo "[dispatch] $name -> GPU $g ($(date +%m/%d\ %H:%M))"
      CUDA_VISIBLE_DEVICES=$g DATA_MODULE=data_mp3d setsid nohup python3 0820_train_oaa_afm.py \
        --run-name $name --audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 \
        --nviews 8 --data-mode r8 --epochs 30 --batch-size 4 --accum 1 --stem-stride1 --seed 1 \
        --lr 5e-4 --out-dir comparison_0820 > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      exit 0
    fi
  done
  [ -e "comparison_0820/logs/$name.log" ] && exit 0
  sleep 180
done
