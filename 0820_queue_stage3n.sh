#!/bin/bash
# 0820 Stage-3n: Beyond-Image-to-Depth (Parida et al., CVPR'21; audio-only port, model/beyond_i2d.py)
# full channel x dataset comparison, mirroring the release baseline recipe (train_baseline fin:
# lr 1e-3, 40 ep, warmup 2; bs32 for the heavy 4-branch model). Empty-GPU-only dispatch.
cd "$(dirname "$0")"
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
JOBS=(
"0820_beyond_fb_rep|data_0422|--mode fb"
"0820_beyond_fb_mp3d|data_mp3d|--mode fb"
"0820_beyond_r2_rep|data_0422|--mode r2"
"0820_beyond_r2_mp3d|data_mp3d|--mode r2"
"0820_beyond_r6_rep|data_0422|--mode r6"
"0820_beyond_r6_mp3d|data_mp3d|--mode r6"
"0820_beyond_r8_rep|data_0422|--mode r8"
"0820_beyond_r8_mp3d|data_mp3d|--mode r8"
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
      CUDA_VISIBLE_DEVICES=$g DATA_MODULE=$dm setsid nohup python3 train_baseline.py \
        --model beyond --run-name $name $extra --lr 1e-3 --warmup-ep 2 --epochs 40 \
        --batch-size 32 --out-dir comparison_0820 \
        > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      i=$((i+1)); sleep 90
    fi
  done
  sleep 180
done
echo "[dispatch] all ${#JOBS[@]} stage-3n jobs launched"
