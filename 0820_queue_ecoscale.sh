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
mkdir -p /tmp/0820_gpulock
# single-instance guard (2026-09-24): a second copy racing on the same JOBS/logs is what
# doubled the [gate] line and the dispatch rate.
exec 8>"/tmp/0820_gpulock/.ecoscale.lock"
flock -n 8 || { echo "[abort] another 0820_queue_ecoscale.sh is already running"; exit 0; }
export MICGAIN_ROOT=/root/storage/supple_mic_gain R0422_SPLIT=off3
# 2026-09-24: out-dir is overridable so a node can keep checkpoints on its LOCAL disk
# (the shared /data is 99% full). Default unchanged -> other nodes behave as before.
# The fairhop GATE above/below deliberately still reads the SHARED comparison_0820/logs.
OUT="${OUT:-comparison_0820}"
ECHODIFF_PY="${ECHODIFF_PY:-/root/local1/changwoo/echodiff_env/bin/python}"
mkdir -p "$OUT/logs" /tmp/0820_gpulock
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; NEED_MB="${NEED_MB:-26000}"; MAX_TRY="${MAX_TRY:-2}"
# 2026-09-24: the retry counter lives on the filesystem (how many $lg.failN exist), NOT in a
# shell array -- pending() runs inside $( ), so assignments to a parent-shell array are discarded
# and MAX_TRY was never reached (141 identical relaunches of the eco-scale jobs).

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
  # 8>&- : the trainer must NOT inherit the single-instance lock fd. flock only releases
  # when the last fd closes, so an inherited fd 8 keeps .ecoscale.lock held for as long as
  # any job lives -- the queue could then never be restarted while work was in flight.
  CUDA_VISIBLE_DEVICES=$1 DATA_MODULE=data_micgain STFT_HOP=44 setsid nohup "$ECHODIFF_PY" \
    train_echodiffusion.py --run-name "$2" --mode "$3" $4 --resume auto --out-dir "$OUT" \
    > "$OUT/logs/$2.log" 2>&1 < /dev/null 8>&- &
  local t=0
  while [ $t -lt 400 ]; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1")
    [ "$free" -lt "$NEED_MB" ] && break
    if grep -qiE "OutOfMemory|Traceback" "$OUT/logs/$2.log" 2>/dev/null; then
      echo "[crash] $2 died in ${t}s: $(grep -m1 -A1 Traceback "$OUT/logs/$2.log" | tail -1)"
      break
    fi
    sleep 10; t=$((t+10))
  done
  exec 9>&-; return 0
}
pending() {
  local k
  for k in "${!JOBS[@]}"; do
    IFS='|' read -r nm _ <<< "${JOBS[$k]}"
    local lg="$OUT/logs/$nm.log"
    if [ ! -e "$lg" ]; then echo "$k"; continue; fi
    grep -q "\[done\]" "$lg" 2>/dev/null && continue
    pgrep -f -- "--run-name $nm " >/dev/null && continue
    local t; t=$(ls -1 "$lg".fail* 2>/dev/null | wc -l)
    if [ "$t" -lt "$MAX_TRY" ]; then
      mv "$lg" "$lg.fail$((t+1))" 2>/dev/null
      echo "[dispatch] retry $nm (attempt $((t+2)))" >&2; echo "$k"
    elif [ ! -e "$lg.gaveup" ]; then
      : > "$lg.gaveup"
      echo "[dispatch] GIVE UP $nm after $t failed attempts -- see $lg.fail*" >&2
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
