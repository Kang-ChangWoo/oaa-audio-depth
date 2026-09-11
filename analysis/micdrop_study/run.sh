#!/bin/bash
# 8->1 mic degradation study: run every roster model through analysis/micdrop.py.
#
#   bash analysis/micdrop_study/run.sh            # fill in whatever is missing
#   GPUS="3 5" bash analysis/micdrop_study/run.sh # restrict to these GPUs
#
# One JSON shard per model (parallel-write safe, resumable: an existing shard is skipped).
# Dispatch waits for an empty GPU (<2000MB) so a shared node is never disturbed.
set -u
cd "$(dirname "$0")/../.."
OUT=analysis/micdrop_study/shards
mkdir -p "$OUT"

export REPLICA_ROOT=${REPLICA_ROOT:-/root/local2/replica_0422_lite}
export R0422_SPLIT=${R0422_SPLIT:-off3}
export DATA_MODULE=${DATA_MODULE:-data_0422}
export EVAL_BS=${EVAL_BS:-6}
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

RUNS=$(python3 -c "
from analysis.micdrop_study.roster import ROSTER
print(' '.join(e[-1] for e in ROSTER))")

for run in $RUNS; do
  shard="$OUT/$run.json"; claim="$OUT/.$run.claim"
  [ -e "$shard" ] && { echo "[skip] $run (shard exists)"; continue; }
  # claim marker: lets several instances of this script share the roster without collisions
  ( set -o noclobber; : > "$claim" ) 2>/dev/null || { echo "[skip] $run (claimed)"; continue; }
  while :; do
    for g in $GPUS; do
      mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
      if [ "$mem" -lt 2000 ]; then
        echo "[run] $run -> GPU $g ($(date +%m/%d\ %H:%M))"
        CUDA_VISIBLE_DEVICES=$g python3 analysis/micdrop.py --run-name "$run" --out "$shard"
        [ -e "$shard" ] || rm -f "$claim"      # failed: let a later pass retry it
        break 2
      fi
    done
    sleep 120
  done
done
# EchoDiffusion lives in its own env (fp32, own deps) and has its own runner with the identical
# protocol and output format, so its shard merges straight into the study.
eco=$(python3 -c "from analysis.micdrop_study.roster import ECO; print(ECO[-1])")
shard="$OUT/$eco.json"; claim="$OUT/.$eco.claim"
if [ -n "${ECHODIFF_PY:-}" ] && [ ! -e "$shard" ] && ( set -o noclobber; : > "$claim" ) 2>/dev/null; then
  while :; do
    for g in $GPUS; do
      mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
      if [ "$mem" -lt 2000 ]; then
        echo "[run] $eco -> GPU $g (echodiff env)"
        CUDA_VISIBLE_DEVICES=$g HF_HOME=${HF_HOME:-/root/local1/changwoo/_afm_weights} \
          "$ECHODIFF_PY" analysis/micdrop_eco.py --run-name "$eco" --out "$shard"
        [ -e "$shard" ] || rm -f "$claim"
        break 2
      fi
    done
    sleep 120
  done
fi

echo "[done] shards in $OUT — build the table with analysis/micdrop_study/report.py"
