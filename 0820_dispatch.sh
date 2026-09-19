#!/bin/bash
# Serialised dispatcher for the additional-experiment programme (2026-09-20).
#
# Priorities follow the request, and every job below was checked against the filesystem first so
# nothing already measured is retrained:
#
#  P1  the ONE missing cell of the current-best 8-cell benchmark. A sweep of comparison_0820 args
#      found seven of the eight sslam+LLRD+conv-stem+hop44 cells already trained AND evaluated;
#      only Replica 2ch is absent. (exp 1)
#  P2  the hop/context ablation on MP3D 4ch. Conditions A (58.3 ms, hop 160 -> 0.7744) and
#      B (58.3 ms, hop 44 -> 0.7562) already exist, so only C and D are new. C is the matched-frame
#      control: the source wavs are much longer than the released crop (Replica 1641 ms, MP3D
#      >= 1147 ms), and 210 ms at hop 160 gives EXACTLY 64 STFT frames -- the same count as 58.3 ms
#      at hop 44 (verified on real tensors, not arithmetic). If C ~ B the gain is about frame count
#      or acoustic context; if B > C it is about sampling the short echo densely. D adds both. (exp 2)
#  P5  multi-seed replication of the result the paper claim rests on (MP3D 8ch hop44) -- already
#      running as 0820_h44_sslcs_r8_mp3d_s1, not repeated here.
#
# Microphone scaling (exp 3) needs a new data module for /root/storage/supple_mic_gain (12 yaw
# headings = up to 24 channels, nested bisection order) and nviews > 8 support in the model; it is
# queued separately once that is written and smoke-tested.
#
#   bash 0820_dispatch.sh
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs /tmp/0820_gpulock
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
# A job needs ~20-27 GB. Earlier dispatchers tested "used < 2000 MB", which says nothing about
# whether a 49 GB card has room for one more; that produced a second wave of OOM deaths on cards
# already holding a run. Require actual FREE memory instead.
NEED_MB="${NEED_MB:-26000}"

SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv"
FBMP="--nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"

#  name | data module | HOP | WINDOW | trainer args
JOBS=(
# --- P1: the missing benchmark cell
"0820_h44_sslcs_r2_rep|data_0422|44|2799|$SL --nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2"
# --- P2: hop / context ablation, MP3D 4ch. A and B already exist.
"0820_ctx_C_210ms_h160_mp3d|data_mp3d|160|10080|$SL $FBMP"
"0820_ctx_D_210ms_h44_mp3d|data_mp3d|44|10080|$SL $FBMP"
# --- P6: the remaining window/hop axes, lowest priority (win 100 already came back clearly worse)
"0820_s9_h88_mp3d|data_mp3d|88|2823|$SL $FBMP"
"0820_s9_h22_mp3d|data_mp3d|22|2823|$SL $FBMP"
)

launch() {   # $1 gpu  $2 name  $3 dm  $4 hop  $5 window  $6 args
  local lock="/tmp/0820_gpulock/$1"
  exec 9>"$lock"; flock -n 9 || return 1
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1")
  [ "$free" -ge "$NEED_MB" ] || { exec 9>&-; return 1; }
  echo "[dispatch] $2 (hop $4 window $5) -> GPU $1 ($(date +%m/%d\ %H:%M))"
  CUDA_VISIBLE_DEVICES=$1 DATA_MODULE=$3 STFT_HOP=$4 STFT_WINDOW=$5 setsid nohup \
    python3 0820_train_oaa_afm.py --run-name "$2" $6 --lr 5e-4 --out-dir comparison_0820 \
    > "comparison_0820/logs/$2.log" 2>&1 < /dev/null &
  local t=0
  while [ $t -lt 300 ]; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1")
    [ "$free" -lt "$NEED_MB" ] && break
    grep -qiE "OutOfMemory|Traceback" "comparison_0820/logs/$2.log" 2>/dev/null && break
    sleep 10; t=$((t+10))
  done
  exec 9>&-
  return 0
}

# Every sweep re-derives the pending set from the filesystem instead of advancing a cursor.
# The previous loop walked an index forward and skipped any job whose log already existed at the
# moment it passed; when a run died of OOM and its log was archived afterwards, that job could
# never be retried and had to be launched by hand (this happened to the Replica 2ch benchmark
# cell). Re-scanning also means a job is picked up automatically if its log is removed, and that
# an OOM death is retried rather than silently lost.
#
# A run is considered SETTLED (never relaunched) when its log records [done]; a log that exists
# without [done] and without a live process is a crashed run, which IS retried, up to MAX_TRY
# attempts, with the dead log archived as .failN so nothing is overwritten.
MAX_TRY="${MAX_TRY:-2}"
declare -A TRIES

pending() {   # echo the indices of jobs still needing a launch
  local k
  for k in "${!JOBS[@]}"; do
    IFS='|' read -r nm _ <<< "${JOBS[$k]}"
    local lg="comparison_0820/logs/$nm.log"
    if [ ! -e "$lg" ]; then echo "$k"; continue; fi
    grep -q "\[done\]" "$lg" 2>/dev/null && continue            # finished
    pgrep -f -- "--run-name $nm " >/dev/null && continue          # still training
    # log exists, no [done], no process -> crashed; retry a bounded number of times
    local t=${TRIES[$nm]:-0}
    if [ "$t" -lt "$MAX_TRY" ]; then
      TRIES[$nm]=$((t+1))
      mv "$lg" "$lg.fail$((t+1))" 2>/dev/null
      echo "[dispatch] retry $nm (attempt $((t+2))/$((MAX_TRY+1))); previous log -> $lg.fail$((t+1))"
      echo "$k"
    fi
  done
}

while true; do
  todo=($(pending))
  [ ${#todo[@]} -eq 0 ] && break
  for g in $GPUS; do
    [ ${#todo[@]} -eq 0 ] && break
    IFS='|' read -r name dm hop win extra <<< "${JOBS[${todo[0]}]}"
    if launch "$g" "$name" "$dm" "$hop" "$win" "$extra"; then todo=("${todo[@]:1}"); fi
  done
  sleep 120
done
echo "[dispatch] all ${#JOBS[@]} jobs settled"
