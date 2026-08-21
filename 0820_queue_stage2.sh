#!/bin/bash
# 0820 Stage-2: hypothesis-testing variants. Dispatches one job per free GPU among 2-7 as the
# screening runs finish. Jobs (1 hypothesis each):
#   randinit  : pretraining contribution (same arch/LR, random init)
#   seed1     : is the sslam~CNN Replica tie seed noise?
#   convstem  : sub-patch locality (4x stride-2 conv patch stem)
#   m2d20ms   : temporal resolution (patch 128x2 -> time-slice tokens), both datasets
#   llrd      : layer-wise LR decay 0.75 + warmup 8 (post-norm spike fix), both datasets
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
# name|DATA_MODULE|extra-args   (Replica first: short jobs fill freed GPUs early)
JOBS=(
"0820_sslam_randinit_fb_rep|data_0422|--audio-backbone sslam --afm-random-init --accum 4"
"0820_sslam_s1_fb_rep|data_0422|--audio-backbone sslam --seed 1 --accum 4"
"0820_sslam_convstem_fb_rep|data_0422|--audio-backbone sslam --afm-stem conv --accum 4"
"0820_m2d20ms_fb_rep|data_0422|--audio-backbone m2d20ms --accum 4"
"0820_sslam_llrd_fb_rep|data_0422|--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 --accum 4"
"0820_m2d20ms_fb_mp3d|data_mp3d|--audio-backbone m2d20ms --accum 1"
"0820_sslam_llrd_fb_mp3d|data_mp3d|--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 --accum 1"
"0820_bat_dbmm_fb_rep|data_0422|--audio-backbone bat --afm-input-norm db_minmax --accum 4"
"0820_audiomosaic_db_fb_rep|data_0422|--audio-backbone audiomosaic --afm-input-norm db --accum 4"
"0820_eat_llrd_fb_mp3d|data_mp3d|--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8 --accum 1"
"0820_bat_dbmm_fb_mp3d|data_mp3d|--audio-backbone bat --afm-input-norm db_minmax --accum 1"
)
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for g in 2 3 4 5 6 7; do
    [ $i -ge ${#JOBS[@]} ] && break
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$mem" -lt 2000 ]; then
      IFS='|' read -r name dm extra <<< "${JOBS[$i]}"
      if [ -e "comparison_0820/logs/$name.log" ]; then i=$((i+1)); continue; fi
      echo "[dispatch] $name -> GPU $g ($(date +%H:%M))"
      CUDA_VISIBLE_DEVICES=$g DATA_MODULE=$dm setsid nohup python3 0820_train_oaa_afm.py \
        --run-name $name $extra --nviews 4 --data-mode fb --lr 5e-4 --epochs 40 --batch-size 8 \
        --out-dir comparison_0820 > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      i=$((i+1)); sleep 60   # let the job claim GPU memory before rechecking
    fi
  done
  sleep 180
done
echo "[dispatch] all ${#JOBS[@]} stage-2 jobs launched"
