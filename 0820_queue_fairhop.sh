#!/bin/bash
# Preprocessing-fairness control: rebuild BOTH baselines at hop 44 (experiment 4, promoted).
#
# WHY THIS IS NOT OPTIONAL. The 8-cell benchmark currently reads our model at hop 44 against
# OAA-CNN and EchoDiffusion at hop 160, which prices the input recipe together with the encoder.
# Where the CNN has already been retrained at hop 44 the picture changes materially:
#
#   cell          CNN@160   CNN@44   ours@44    vs CNN@160   vs CNN@44
#   Replica 4ch    0.2596   0.2496    0.2496      -0.0100      0.0000   tie
#   Replica 8ch    0.2368   0.2350    0.2371      +0.0003     +0.0021   tie
#   MP3D 4ch       0.7849   0.7744    0.7562      -0.0288     -0.0182   win
#
# So on Replica the win against the released CNN is the INPUT, not the encoder; on MP3D a real
# margin survives. Four CNN cells and all eight EchoDiffusion cells are still missing at hop 44,
# and until they exist the headline table cannot be read as an encoder comparison.
#
# CNN first: it is cheaper, it is our own baseline, and it completes the matched comparison for the
# model the paper's claim is stated against. EchoDiffusion runs in its own env ($ECHODIFF_PY) and
# consumes both the spec and the waveform, so hop 44 reaches it through the same data module with
# no code change -- only STFT_HOP is set. MP3D 8ch CNN is already being retrained as
# 0820_h44_cnn_r8_mp3d_s1 (the first attempt never learned) and is not repeated here.
#
#   ECHODIFF_PY=/root/local1/changwoo/echodiff_env/bin/python bash 0820_queue_fairhop.sh
cd "$(dirname "$0")"
mkdir -p /tmp/0820_gpulock
# single-instance guard (2026-09-24), same reason as 0820_queue_ecoscale.sh
exec 8>"/tmp/0820_gpulock/.fairhop.lock"
flock -n 8 || { echo "[abort] another 0820_queue_fairhop.sh is already running"; exit 0; }
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
ECHODIFF_PY="${ECHODIFF_PY:-/root/local1/changwoo/echodiff_env/bin/python}"
mkdir -p comparison_0820/logs /tmp/0820_gpulock
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
NEED_MB="${NEED_MB:-26000}"
MAX_TRY="${MAX_TRY:-2}"
# 2026-09-24: the retry counter lives on the filesystem (how many $lg.failN exist), NOT in a
# shell array -- pending() runs inside $( ), so assignments to a parent-shell array are discarded
# and MAX_TRY was never reached (141 identical relaunches of the eco-scale jobs).

CN="--audio-backbone cnn --warmup-ep 4"
#  name | kind | data module | mode | extra
JOBS=(
# --- OAA-CNN at hop 44, the four cells still missing
"0820_h44_cnn_r2_rep|oaa|data_0422|r2|$CN --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2"
"0820_h44_cnn_r6_rep|oaa|data_0422|r6|$CN --nviews 6 --data-mode r6 --epochs 40 --batch-size 4 --accum 8"
"0820_h44_cnn_r2_mp3d|oaa|data_mp3d|r2|$CN --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2"
"0820_h44_cnn_r6_mp3d|oaa|data_mp3d|r6|$CN --nviews 6 --data-mode r6 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"
# --- EchoDiffusion at hop 44, all eight cells
"0820_h44_eco_r2_rep|eco|data_0422|r2|--epochs 40 --batch-size 12"
"0820_h44_eco_fb_rep|eco|data_0422|fb|--epochs 40 --batch-size 8"
"0820_h44_eco_r6_rep|eco|data_0422|r6|--epochs 40 --batch-size 4"
"0820_h44_eco_r8_rep|eco|data_0422|r8|--epochs 40 --batch-size 3"
"0820_h44_eco_r2_mp3d|eco|data_mp3d|r2|--epochs 40 --batch-size 12"
"0820_h44_eco_fb_mp3d|eco|data_mp3d|fb|--epochs 40 --batch-size 8"
"0820_h44_eco_r6_mp3d|eco|data_mp3d|r6|--epochs 30 --batch-size 4"
"0820_h44_eco_r8_mp3d|eco|data_mp3d|r8|--epochs 30 --batch-size 3"
)

launch() {   # $1 gpu  $2 name  $3 kind  $4 dm  $5 mode  $6 args
  exec 9>"/tmp/0820_gpulock/$1"; flock -n 9 || return 1
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1")
  [ "$free" -ge "$NEED_MB" ] || { exec 9>&-; return 1; }
  echo "[dispatch] $2 ($3, hop 44) -> GPU $1 ($(date +%m/%d\ %H:%M))"
  if [ "$3" = "eco" ]; then
    CUDA_VISIBLE_DEVICES=$1 DATA_MODULE=$4 STFT_HOP=44 setsid nohup "$ECHODIFF_PY" \
      train_echodiffusion.py --run-name "$2" --mode "$5" $6 --out-dir comparison_0820 \
      > "comparison_0820/logs/$2.log" 2>&1 < /dev/null &
  else
    CUDA_VISIBLE_DEVICES=$1 DATA_MODULE=$4 STFT_HOP=44 setsid nohup python3 \
      0820_train_oaa_afm.py --run-name "$2" $6 --lr 5e-4 --out-dir comparison_0820 \
      > "comparison_0820/logs/$2.log" 2>&1 < /dev/null &
  fi
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
    IFS='|' read -r name kind dm mode extra <<< "${JOBS[${todo[0]}]}"
    if launch "$g" "$name" "$kind" "$dm" "$mode" "$extra"; then todo=("${todo[@]:1}"); fi
  done
  sleep 120
done
echo "[dispatch] all ${#JOBS[@]} fairness jobs settled"
