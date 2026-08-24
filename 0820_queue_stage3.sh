#!/bin/bash
# 0820 Stage-3: confirmation + evidence-based variants. Free-GPU dispatcher (GPUs 2-7).
#  R1 eatllrd_fb_rep        : winner recipe on Replica (both-datasets claim)
#  R2/R3 eat_llrd s1/s2 MP3D: multi-seed of the CNN-beating run
#  V1 m2d20ms_llrd MP3D     : combine the two validated partial gains (time-res + LLRD)
#  V2 eatllrd_e30 MP3D      : 30ep cosine matched to the observed ep13 minimum (deeper min?)
#  V3 m2d_llrd MP3D         : does LLRD generalize to prenorm backbones (no-spike case)?
#  V4 m2d20ms_llrd_fb_rep   : combo on Replica (fills the GPU freed by R1)
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
JOBS=(
"0820_eatllrd_fb_rep|data_0422|--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8 --epochs 40 --accum 4"
"0820_eat_llrd_fb_mp3d_s1|data_mp3d|--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8 --epochs 40 --seed 1 --accum 1"
"0820_eat_llrd_fb_mp3d_s2|data_mp3d|--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8 --epochs 40 --seed 2 --accum 1"
"0820_m2d20ms_llrd_fb_mp3d|data_mp3d|--audio-backbone m2d20ms --afm-llrd 0.75 --warmup-ep 8 --epochs 40 --accum 1"
"0820_eatllrd_e30_fb_mp3d|data_mp3d|--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8 --epochs 30 --accum 1"
"0820_m2d_llrd_fb_mp3d|data_mp3d|--audio-backbone m2d --afm-llrd 0.75 --warmup-ep 8 --epochs 40 --accum 1"
"0820_m2d20ms_llrd_fb_rep|data_0422|--audio-backbone m2d20ms --afm-llrd 0.75 --warmup-ep 8 --epochs 40 --accum 4"
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
        --run-name $name $extra --nviews 4 --data-mode fb --lr 5e-4 --batch-size 8 \
        --out-dir comparison_0820 > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      i=$((i+1)); sleep 60
    fi
  done
  sleep 180
done
echo "[dispatch] all ${#JOBS[@]} stage-3 jobs launched"
