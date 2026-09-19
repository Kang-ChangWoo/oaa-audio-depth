#!/bin/bash
# Single serialised dispatcher for every remaining 0820 queue.
#
# Why this replaces the per-stage dispatchers. Each stage script ran its own loop that (a) polled
# for a GPU under 2000 MB, then (b) slept 90 s before launching. With two stage loops alive at once
# the same GPU passed both checks and took two jobs, and five runs died of CUDA OOM in a row. This
# script holds a lock file per GPU so only one launch can claim a device, verifies the device is
# still free immediately before exec, and waits for the new process to actually allocate before
# considering the slot taken.
#
# Job list is ordered by what the analysis needs next, not by which stage produced it:
#   1. stage-10 path separation  -- the only open explanation for the hop-44 gain now that
#      information content, token allocation and time resolution have all been ruled out.
#   2. stage-9 token control     -- finishes the grid that ruled token allocation out.
#   3. stage-9 window / hop      -- the remaining axes, lowest priority since win 100 at hop 160
#      already came back clearly worse (0.3872 vs 0.2927).
#
#   bash 0820_dispatch.sh
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs /tmp/0820_gpulock
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
FREE_MB="${FREE_MB:-2000}"

SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv"
FB="--nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"

#  name | data module | HOP | WIN | trainer args
JOBS=(
# --- 1. path separation (stage 10)
"0820_s10_afm18_rep|data_0422|44|400|$SL $FB --afm-frames 18"
"0820_s10_fine18_mp3d|data_mp3d|44|400|$SL $FB --fine-frames 18"
# --- 2. token allocation, remaining cell
"0820_s9_tok328_h44_rep|data_0422|44|400|$SL $FB --afm-patch 32,8"
# --- 3. window and hop axes
"0820_s9_w100_h44_rep|data_0422|44|100|$SL $FB"
"0820_s9_w200_h44_mp3d|data_mp3d|44|200|$SL $FB"
"0820_s9_w100_h160_mp3d|data_mp3d|160|100|$SL $FB"
"0820_s9_w200_h160_mp3d|data_mp3d|160|200|$SL $FB"
"0820_s9_w100_h44_mp3d|data_mp3d|44|100|$SL $FB"
"0820_s9_h88_mp3d|data_mp3d|88|400|$SL $FB"
"0820_s9_h22_mp3d|data_mp3d|22|400|$SL $FB"
"0820_s9_h88_rep|data_0422|88|400|$SL $FB"
"0820_s9_h22_rep|data_0422|22|400|$SL $FB"
)

launch() {   # $1 gpu  $2 name  $3 dm  $4 hop  $5 win  $6 args
  local lock="/tmp/0820_gpulock/$1"
  exec 9>"$lock"; flock -n 9 || return 1                       # one claim per device
  local mem; mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1")
  [ "$mem" -lt "$FREE_MB" ] || { exec 9>&-; return 1; }         # re-check immediately before exec
  echo "[dispatch] $2 (hop $4 win $5) -> GPU $1 ($(date +%m/%d\ %H:%M))"
  CUDA_VISIBLE_DEVICES=$1 DATA_MODULE=$3 STFT_HOP=$4 STFT_WIN=$5 setsid nohup \
    python3 0820_train_oaa_afm.py --run-name "$2" $6 --lr 5e-4 --out-dir comparison_0820 \
    > "comparison_0820/logs/$2.log" 2>&1 < /dev/null &
  local t=0                                                     # hold the lock until it allocates
  while [ $t -lt 300 ]; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1")
    [ "$mem" -ge "$FREE_MB" ] && break
    grep -qiE "OutOfMemory|Traceback" "comparison_0820/logs/$2.log" 2>/dev/null && break
    sleep 10; t=$((t+10))
  done
  exec 9>&-
  return 0
}

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for g in $GPUS; do
    [ $i -ge ${#JOBS[@]} ] && break
    IFS='|' read -r name dm hop win extra <<< "${JOBS[$i]}"
    if [ -e "comparison_0820/logs/$name.log" ]; then i=$((i+1)); continue; fi
    launch "$g" "$name" "$dm" "$hop" "$win" "$extra" && i=$((i+1))
  done
  sleep 120
done
echo "[dispatch] all ${#JOBS[@]} jobs launched"
