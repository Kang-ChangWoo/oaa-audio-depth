#!/bin/bash
# 0820 Stage-3j: the unified-winner push — sslam as the single setting across channels x datasets.
#  1) sslam_r6_rep s1/s2      : is the r6 full-metric sweep (0.2271, -4.7% vs CNN) seed-robust?
#  2) sslam_llrd_e30_fb_mp3d  : close the MP3D fb gap (0.792 vs CNN 0.785) with the e30 cosine that helped eat
#  3) sslam_llrd65_fb_mp3d    : stronger layer decay (0.65) — preserve more pretrained features on noisy MP3D
#  4) sslam_llrd_r6_rep       : does the unified recipe (sslam+LLRD) keep the Replica r6 win?
#  5) sslam_llrd_r2_mp3d      : AFM-favorable low-channel MP3D cell with the unified recipe
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8"
SEEDS="${SEEDS:-0}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
JOBS=(
"0820_sslam_s2_fb_rep|data_0422|--audio-backbone sslam --nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4 --seed 2"
"0820_sslam_llrd65_fb_mp3d_s1|data_mp3d|--audio-backbone sslam --afm-llrd 0.65 --warmup-ep 8 --nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4 --seed 1"
"0820_sslam_llrd65_fb_mp3d_s2|data_mp3d|--audio-backbone sslam --afm-llrd 0.65 --warmup-ep 8 --nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4 --seed 2"
"0820_eatllrd_cs_r6_mp3d|data_mp3d|--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv --nviews 6 --data-mode r6 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"
"0820_eatllrd_cs_r8_mp3d|data_mp3d|--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv --nviews 8 --data-mode r8 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"
"0820_eatllrd_cs_r2_mp3d|data_mp3d|--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2"
"0820_eatllrd_cs_r2_rep|data_0422|--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2"
"0820_eatllrd_cs_r6_rep|data_0422|--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv --nviews 6 --data-mode r6 --epochs 40 --batch-size 4 --accum 8"
"0820_eatllrd_cs_r8_rep|data_0422|--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv --nviews 8 --data-mode r8 --epochs 40 --batch-size 3 --accum 11 --subset-aug --vdrop-kmax 4"
)
SJOBS=()
for j in "${JOBS[@]}"; do SJOBS+=("$j"); done
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
echo "[dispatch] all ${#SJOBS[@]} stage-3j jobs launched"
