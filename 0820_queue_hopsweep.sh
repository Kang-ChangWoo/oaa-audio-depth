#!/bin/bash
# hop sweep on one representative cell, so the final hop is chosen on a defensible criterion
# rather than on our own model's best score.
#
# THE PROBLEM WITH 44. It is not a conventional STFT hop and it was reached by trying values, so a
# reader can fairly say it was tuned to us. A hop has to be defensible without reference to our
# score. The measured structure of this task offers two candidates that are:
#
#   hop 128 = n_fft/4   the standard choice in the audio literature (COLA / Griffin-Lim default).
#                       Gives 22 real frames against 32 time tokens -> the grid is UNDERFILLED.
#   hop  88             gives exactly 32 frames = exactly the model's 32 time tokens, one real
#                       frame per token: no replication, no oversampling. Structural, not tuned.
#   hop  64 = n_fft/8   a power-of-two divisor, 44 frames, grid comfortably filled.
#   hop  44             current experiment value, 64 frames = 2x the tokens (each token averages 2).
#
# Running all of them on one cell lets the choice be stated as "the coarsest hop that fills the
# model's time-token grid" (a property of the architecture, not of our accuracy) and lets us report
# how much, if anything, is lost by moving from 44 to that value. MP3D 4ch is the representative
# cell: it has hop 160 (0.7744) and hop 44 (0.7562) already measured, so the sweep has both anchors.
#
# hop 88 and 22 are already queued elsewhere; only 128 and 64 are added here.
#
#   bash 0820_queue_hopsweep.sh
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs /tmp/0820_gpulock
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; NEED_MB="${NEED_MB:-26000}"; MAX_TRY="${MAX_TRY:-2}"
declare -A TRIES
SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv"
FB="--nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"

JOBS=(
"0820_s9_h128_mp3d|128|$SL $FB"
"0820_s9_h64_mp3d|64|$SL $FB"
)

launch() {
  exec 9>"/tmp/0820_gpulock/$1"; flock -n 9 || return 1
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1")
  [ "$free" -ge "$NEED_MB" ] || { exec 9>&-; return 1; }
  echo "[dispatch] $2 (hop $3) -> GPU $1 ($(date +%m/%d\ %H:%M))"
  CUDA_VISIBLE_DEVICES=$1 DATA_MODULE=data_mp3d STFT_HOP=$3 setsid nohup python3 \
    0820_train_oaa_afm.py --run-name "$2" $4 --lr 5e-4 --out-dir comparison_0820 \
    > "comparison_0820/logs/$2.log" 2>&1 < /dev/null &
  local t=0
  while [ $t -lt 400 ]; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1")
    [ "$free" -lt "$NEED_MB" ] && break
    grep -qiE "OutOfMemory|Traceback" "comparison_0820/logs/$2.log" 2>/dev/null && break
    sleep 10; t=$((t+10))
  done
  exec 9>&-; return 0
}
pending() {
  for k in "${!JOBS[@]}"; do
    IFS='|' read -r nm _ <<< "${JOBS[$k]}"; lg="comparison_0820/logs/$nm.log"
    [ ! -e "$lg" ] && { echo "$k"; continue; }
    grep -q "\[done\]" "$lg" 2>/dev/null && continue
    pgrep -f -- "--run-name $nm " >/dev/null && continue
    t=${TRIES[$nm]:-0}
    [ "$t" -lt "$MAX_TRY" ] && { TRIES[$nm]=$((t+1)); mv "$lg" "$lg.fail$((t+1))" 2>/dev/null; echo "$k"; }
  done
}
while true; do
  todo=($(pending)); [ ${#todo[@]} -eq 0 ] && break
  for g in $GPUS; do
    [ ${#todo[@]} -eq 0 ] && break
    IFS='|' read -r name hop extra <<< "${JOBS[${todo[0]}]}"
    launch "$g" "$name" "$hop" "$extra" && todo=("${todo[@]:1}")
  done
  sleep 120
done
echo "[dispatch] hop sweep settled"
