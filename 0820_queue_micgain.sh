#!/bin/bash
# Microphone-scaling study on /root/storage/supple_mic_gain (experiment 3).
#
# DATASET, as read from the set itself rather than assumed. mic_order.json records a 30-degree yaw
# grid (not 45), so there are 12 binaural headings and N_max = 24 channels, not 8. Headings are
# added in a bisection order -- 0, 180, 90, 270, 30, 210, 120, 300, 60, 240, 150, 330 -- each new
# yaw falling in the largest remaining gap, so the first k entries are the k-heading set and the
# subsets are NESTED by construction (verified in data_micgain: every mode's channel list is a
# prefix of the next). k=4 reproduces exactly the released r8 input {0, 90, 180, 270}.
#
# The set's README records that the renderer has no RNG seed, so yaw 0 here is not bit-identical to
# replica_0422 (waveform correlation 0.82-0.98 away from position 0). The scaling comparison is
# therefore closed inside this set: the m8 run below is trained here rather than borrowed from the
# campaign, so the curve is not contaminated by renderer noise.
#
# CONTROLLED VARIABLE. Only the channel count changes. Recipe, fusion, decoder, loss, optimiser,
# checkpoint rule and hop are the campaign's current best (sslam + LLRD + conv stem + hop 44), and
# the parameter count is identical at every channel count (100,713,770, verified 2 through 24).
# Batch size falls with channel count to hold the effective batch near 24 and avoid the OOM that
# killed five runs earlier in this campaign.
#
# Dispatch order is chosen so the curve becomes readable early: the two anchors the analysis needs
# first (m8, matching the released rig, and m24, the maximum) lead, then the ends, then the middle.
#
#   bash 0820_queue_micgain.sh
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export MICGAIN_ROOT=/root/storage/supple_mic_gain R0422_SPLIT=off3
mkdir -p comparison_0820/logs /tmp/0820_gpulock
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
# A job needs ~20-27 GB. Earlier dispatchers tested "used < 2000 MB", which says nothing about
# whether a 49 GB card has room for one more; that produced a second wave of OOM deaths on cards
# already holding a run. Require actual FREE memory instead.
NEED_MB="${NEED_MB:-26000}"
SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv --epochs 40"

#  channels | batch | accum        (effective batch ~24 throughout)
SCHED=( "8|3|8" "24|1|24" "2|12|2" "4|8|3" "6|4|6" "12|2|12" "16|2|12" "20|1|24" "10|2|12" "14|2|12" "18|1|24" "22|1|24" )

launch() {   # $1 gpu  $2 name  $3 args
  local lock="/tmp/0820_gpulock/$1"
  exec 9>"$lock"; flock -n 9 || return 1
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1")
  [ "$free" -ge "$NEED_MB" ] || { exec 9>&-; return 1; }
  echo "[dispatch] $2 -> GPU $1 ($(date +%m/%d\ %H:%M))"
  CUDA_VISIBLE_DEVICES=$1 DATA_MODULE=data_micgain STFT_HOP=44 setsid nohup \
    python3 0820_train_oaa_afm.py --run-name "$2" $3 --lr 5e-4 --out-dir comparison_0820 \
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

i=0
while [ $i -lt ${#SCHED[@]} ]; do
  for g in $GPUS; do
    [ $i -ge ${#SCHED[@]} ] && break
    IFS='|' read -r ch bs acc <<< "${SCHED[$i]}"
    name="0820_mg_ours_m${ch}"
    if [ -e "comparison_0820/logs/$name.log" ]; then i=$((i+1)); continue; fi
    launch "$g" "$name" "$SL --nviews $ch --data-mode m$ch --batch-size $bs --accum $acc" && i=$((i+1))
  done
  sleep 120
done
echo "[dispatch] all ${#SCHED[@]} micgain jobs launched"
