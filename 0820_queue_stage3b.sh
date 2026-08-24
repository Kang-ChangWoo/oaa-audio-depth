#!/bin/bash
# 0820 Stage-3b: carryover + eat_llrd channel scaling 2/6/8 obs on both datasets
# (per-channel fin recipes; r2 bs24 -> bs12 x accum2, identical effective batch, ViT memory).
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
LL="--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8"
JOBS=(
"0820_m2d20ms_llrd_fb_rep|data_0422|--audio-backbone m2d20ms --afm-llrd 0.75 --warmup-ep 8 --nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"
"0820_eatllrd_r2_rep|data_0422|$LL --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2"
"0820_eatllrd_r6_rep|data_0422|$LL --nviews 6 --data-mode r6 --epochs 40 --batch-size 4 --accum 8"
"0820_eatllrd_r8_rep|data_0422|$LL --nviews 8 --data-mode r8 --epochs 40 --batch-size 3 --accum 11 --subset-aug --vdrop-kmax 4"
"0820_eatllrd_r2_mp3d|data_mp3d|$LL --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2"
"0820_eatllrd_r6_mp3d|data_mp3d|$LL --nviews 6 --data-mode r6 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"
"0820_eatllrd_r8_mp3d|data_mp3d|$LL --nviews 8 --data-mode r8 --epochs 30 --batch-size 4 --accum 1 --stem-stride1 --subset-aug --vdrop-kmax 6 --vdrop-kstep 1"
)
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for g in 2 3 4 5 6 7; do
    [ $i -ge ${#JOBS[@]} ] && break
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$mem" -lt 2000 ]; then
      IFS='|' read -r name dm extra <<< "${JOBS[$i]}"
      if [ -e "comparison_0820/logs/$name.log" ]; then i=$((i+1)); continue; fi
      echo "[dispatch] $name -> GPU $g ($(date +%m/%d\ %H:%M))"
      CUDA_VISIBLE_DEVICES=$g DATA_MODULE=$dm setsid nohup python3 0820_train_oaa_afm.py \
        --run-name $name $extra --lr 5e-4 --out-dir comparison_0820 \
        > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      i=$((i+1)); sleep 90   # r2 runs claim memory slower; wait before rechecking
    fi
  done
  sleep 180
done
echo "[dispatch] all ${#JOBS[@]} stage-3b jobs launched"
