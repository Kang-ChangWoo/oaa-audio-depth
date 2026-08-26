#!/bin/bash
# 0820 Stage-3d: no-vdrop r8 controls for eat_llrd (symmetry ablation: does subset_aug help the AFM at r8?)
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
LL="--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8"
SEEDS="${SEEDS:-0}"
GPUS="${GPUS:-2 3 4 5 6 7}"   # e.g. GPUS="0 1 2 3 4 5 6 7"   # multi-seed: SEEDS="0 1 2" bash 0820_queue_stage3d.sh  (seed 0 keeps the base run name)
JOBS=(
"0820_eatllrd_r8novd_rep|data_0422|$LL --nviews 8 --data-mode r8 --epochs 40 --batch-size 3 --accum 11"
"0820_eatllrd_r8novd_mp3d|data_mp3d|$LL --nviews 8 --data-mode r8 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"
)
SJOBS=()
for s in $SEEDS; do
  for j in "${JOBS[@]}"; do
    IFS='|' read -r name dm extra <<< "$j"
    if [ "$s" = "0" ]; then SJOBS+=("$name|$dm|$extra --seed 0");
    else SJOBS+=("${name}_s$s|$dm|$extra --seed $s"); fi
  done
done
i=0
while [ $i -lt ${#SJOBS[@]} ]; do
  for g in $GPUS; do
    [ $i -ge ${#SJOBS[@]} ] && break
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$mem" -lt 2000 ]; then
      IFS='|' read -r name dm extra <<< "${SJOBS[$i]}"
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
echo "[dispatch] all ${#SJOBS[@]} stage-3d jobs launched"
