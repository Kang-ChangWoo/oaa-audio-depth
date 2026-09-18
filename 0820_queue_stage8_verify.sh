#!/bin/bash
# 0820 Stage-8: verify the two runs the hop-44 headline rests on.
#
# Why. MP3D 8ch at hop 44 is the campaign's biggest single result (the only rule-level loss turns
# into a win: 0.9723 -> 0.7275, past OAA-CNN's 0.7467). Two things make it unsafe to publish as is:
#
#  1. THE CONTROL FAILED. 0820_h44_cnn_r8_mp3d never learned -- val sat at 1.21-1.31 for all 30
#     epochs and the loss barely moved (0.0847 -> 0.0830). Its test 0.9956 is a dead run, not
#     evidence that hop 44 hurts the CNN. The same CNN trains normally at hop 44 on MP3D 4ch
#     (0.8831) and Replica 8ch (0.2611), so this is specific to that one run. Without a working
#     control the AFM number cannot be attributed to the encoder rather than the input recipe.
#
#  2. THE WINNING RUN IS UNSTABLE. 0820_h44_sslcs_r8_mp3d reaches 0.8392 val at ep8 and then falls
#     back to ~1.14-1.16 for the remaining 20 epochs -- the same return-to-collapse-basin seen with
#     the B2 gate. The reported 0.7275 comes from a best checkpoint the run itself abandoned, so
#     seed 1 has to show the descent is reproducible and not a lucky epoch.
#
# Both runs are seed 1 of the exact recipes above; nothing else changes.
#
#   bash 0820_queue_stage8_verify.sh
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
R8MP="--nviews 8 --data-mode r8 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"

JOBS=(
"0820_h44_cnn_r8_mp3d_s1|data_mp3d|--audio-backbone cnn --warmup-ep 4 $R8MP --seed 1"
"0820_h44_sslcs_r8_mp3d_s1|data_mp3d|--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv $R8MP --seed 1"
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
      CUDA_VISIBLE_DEVICES=$g DATA_MODULE=$dm STFT_HOP=44 setsid nohup python3 0820_train_oaa_afm.py \
        --run-name $name $extra --lr 5e-4 --out-dir comparison_0820 \
        > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      i=$((i+1)); sleep 90
    fi
  done
  sleep 180
done
echo "[dispatch] all ${#JOBS[@]} stage-8 verification jobs launched"
