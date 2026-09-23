#!/bin/bash
# EchoDiffusion microphone-scaling queue (experiment 3, priority 4): the 12 eco points that close
# the ours-vs-eco scaling comparison on supple_mic_gain. Deliberately NOT started until fairhop has
# PLACED all 12 of its jobs (gate below) — HANDOFF priority: sweep -> fairhop -> micgain -> eco.
#
# Same rules as the ours curve: closed inside this dataset, STFT hop 44 for every point (user
# decision 2026-09-20: scaling stays at 44 regardless of the benchmark hop), only the channel
# count varies. Launcher and retry logic inherited from 0820_queue_fairhop.sh.
#
#   ECHODIFF_PY=/root/local1/changwoo/echodiff_env/bin/python bash 0820_queue_ecoscale.sh
cd "$(dirname "$0")"
export MICGAIN_ROOT=/root/storage/supple_mic_gain R0422_SPLIT=off3
ECHODIFF_PY="${ECHODIFF_PY:-/root/local1/changwoo/echodiff_env/bin/python}"
mkdir -p comparison_0820/logs /tmp/0820_gpulock
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; NEED_MB="${NEED_MB:-26000}"; MAX_TRY="${MAX_TRY:-2}"
declare -A TRIES

# GATE: wait until every fairhop job has a log (i.e. has been dispatched at least once)
FAIRHOP="0820_h44_cnn_r2_rep 0820_h44_cnn_r6_rep 0820_h44_cnn_r2_mp3d 0820_h44_cnn_r6_mp3d 0820_h44_eco_r2_rep 0820_h44_eco_fb_rep 0820_h44_eco_r6_rep 0820_h44_eco_r8_rep 0820_h44_eco_r2_mp3d 0820_h44_eco_fb_mp3d 0820_h44_eco_r6_mp3d 0820_h44_eco_r8_mp3d"
while true; do
  ok=1; for f in $FAIRHOP; do [ -e "comparison_0820/logs/$f.log" ] || { ok=0; break; }; done
  [ $ok -eq 1 ] && break; sleep 300
done
echo "[gate] fairhop fully placed — eco scaling queue opens ($(date +%m/%d\ %H:%M))"

# anchors first (m8 = the released r8 geometry, m24 = the ceiling), then bisection order
JOBS=()
for spec in 8:3 24:1 2:12 4:8 6:4 12:2 16:2 20:1 10:2 14:2 18:1 22:1; do
  IFS=':' read -r ch bs <<< "$spec"
  JOBS+=("0820_mg_eco_m${ch}|m${ch}|--epochs 40 --batch-size $bs")
done

launch() {   # $1 gpu  $2 name  $3 mode  $4 args
  exec 9>"/tmp/0820_gpulock/$1"; flock -n 9 || return 1
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1")
  [ "$free" -ge "$NEED_MB" ] || { exec 9>&-; return 1; }
  echo "[dispatch] $2 (eco-scale, hop 44) -> GPU $1 ($(date +%m/%d\ %H:%M))"
  CUDA_VISIBLE_DEVICES=$1 DATA_MODULE=data_micgain STFT_HOP=44 setsid nohup "$ECHODIFF_PY" \
    train_echodiffusion.py --run-name "$2" --mode "$3" $4 --out-dir comparison_0820 \
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
  local k
  for k in "${!JOBS[@]}"; do
    IFS='|' read -r nm _ <<< "${JOBS[$k]}"
    local lg="comparison_0820/logs/$nm.log"
    if [ ! -e "$lg" ]; then echo "$k"; continue; fi
    grep -q "\[done\]" "$lg" 2>/dev/null && continue
    pgrep -f -- "--run-name $nm " >/dev/null && continue
    local t=${TRIES[$nm]:-0}
    if [ "$t" -lt "$MAX_TRY" ]; then
      TRIES[$nm]=$((t+1)); mv "$lg" "$lg.fail$((t+1))" 2>/dev/null
      echo "[dispatch] retry $nm (attempt $((t+2)))" >&2; echo "$k"
    fi
  done
}
while true; do
  todo=($(pending))
  [ ${#todo[@]} -eq 0 ] && break
  for g in $GPUS; do
    [ ${#todo[@]} -eq 0 ] && break
    IFS='|' read -r name mode extra <<< "${JOBS[${todo[0]}]}"
    launch "$g" "$name" "$mode" "$extra" && todo=("${todo[@]:1}")
  done
  sleep 120
done
echo "[dispatch] eco scaling settled"
