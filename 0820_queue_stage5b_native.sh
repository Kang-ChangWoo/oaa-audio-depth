#!/bin/bash
# 0820 Stage-5b: does the native input layout SUBSUME ConvStem?
#
# ConvStem exists only as a workaround. Its job is to replace the AFM's patch embedding with
# something better than a randomly-initialised linear one -- but the checkpoints ship a pretrained
# patch embedding we were discarding. --afm-stem native stops discarding it, which means there is
# nothing left for ConvStem to fix: the input layer is not modified at all.
#
# If that reading is right, native should reproduce ConvStem's single biggest win without any stem
# engineering. That win is NOT Replica 4ch (where native beat cs by 0.0022, a tie) -- it is the MP3D
# 8ch collapse cell, where ConvStem was worth -0.2484 on eat and took the encoder from cos = 1.0000
# (all eight observations identical) to 0.796. So the decisive cells are the collapse cells, and
# they are what this queue runs, at the released hop so the numbers are directly comparable to the
# existing rows:
#
#   MP3D 8ch   linear stem  sslam+LLRD      0.9861   (partial collapse: dratio 0.285 / cos 0.890)
#              linear stem  sslam            0.9782   (total collapse:   dratio 0.001 / cos 1.000)
#              conv stem    sslam+LLRD+cs    0.9723   (encoder healthy:  dratio 0.429 / cos 0.778)
#              conv stem    eat+LLRD+cs      0.7373   (the only escape)
#              native       <- this queue
#   MP3D 6ch   conv stem    sslam+LLRD+cs    0.7473   (the cell ConvStem won by -0.0375)
#              native       <- this queue
#
# Both get analysis/diff_stats.py afterwards: if native holds cos well below 1.0 with NO stem
# engineering, ConvStem is a workaround we can drop rather than a component we need.
#
#   bash 0820_queue_stage5b_native.sh
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 --afm-stem native"
R8MP="--nviews 8 --data-mode r8 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"
R6MP="--nviews 6 --data-mode r6 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"

JOBS=(
"0820_sslnat_r8_mp3d|data_mp3d|160|$SL $R8MP"
"0820_sslnat_r6_mp3d|data_mp3d|160|$SL $R6MP"
)

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for g in $GPUS; do
    [ $i -ge ${#JOBS[@]} ] && break
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$mem" -lt 2000 ]; then
      IFS='|' read -r name dm hop extra <<< "${JOBS[$i]}"
      if [ -e "comparison_0820/logs/$name.log" ]; then i=$((i+1)); continue; fi
      echo "[dispatch] $name (hop $hop) -> GPU $g ($(date +%m/%d\ %H:%M))"
      CUDA_VISIBLE_DEVICES=$g DATA_MODULE=$dm STFT_HOP=$hop setsid nohup python3 0820_train_oaa_afm.py \
        --run-name $name $extra --lr 5e-4 --out-dir comparison_0820 \
        > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      i=$((i+1)); sleep 90
    fi
  done
  sleep 180
done
echo "[dispatch] all ${#JOBS[@]} stage-5b native jobs launched"
