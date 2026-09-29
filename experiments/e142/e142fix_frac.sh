#!/usr/bin/env bash
# E142-fix sensitivity: how much does the fixed-reference slice-A result depend on the top-FRAC rule?
# Slice A cells only (r8 N=2, r2 N=8) x seeds 0/1/2 x FRAC {0.25, 0.50, 1.00}.
# FRAC=0.25 is re-run into a separate dir to PROVE the --frac edit is inert (must match the grid bit-for-bit).
set -uo pipefail

CODE=/root/storage/e142fix_code
SUBSET=/root/storage/e142_code/e142_make_subset.py
OUT=/root/local1/changwoo/e142fix
REF=$OUT/ref_fixed
PY=${E142_PY:-/opt/conda/bin/python}
JOBS=${JOBS:-6}

export PYTHONPATH=/root/storage/implementation/shared_audio/EchoRecon/src
export PYTHONDONTWRITEBYTECODE=1
MAIN="$OUT/logs/frac_sweep.log"
mkdir -p "$OUT/frac" "$OUT/logs"
echo "=== $(date -u) FRAC sweep start ===" >> "$MAIN"

one() {  # mode N seed frac
  local MODE=$1 N=$2 SEED=$3 FR=$4
  local TAG="f${FR}"
  local CELL="${MODE}_N${N}_s${SEED}"
  local LOG="$OUT/logs/frac_${TAG}_${CELL}.log"
  local DEST="$OUT/frac/${TAG}"
  mkdir -p "$DEST"
  [ -f "$DEST/${CELL}.json" ] && { echo "$(date -u) [skip] $TAG $CELL" >> "$LOG"; return 0; }
  "$PY" "$SUBSET" --mode "$MODE" --nviews "$N" --seed "$SEED" \
        --out-root "$OUT/pred_frac_${TAG}" >> "$LOG" 2>&1 || return 1
  "$PY" "$CODE/e142_eval_fixedref.py" --pred-dir "$OUT/pred_frac_${TAG}/${CELL}" \
        --ref-root "$REF" --frac "$FR" --out "$DEST/${CELL}.json" >> "$LOG" 2>&1
  local RC=$?
  echo "$(date -u) [done] $TAG $CELL rc=$RC" >> "$LOG"
  [ $RC -eq 0 ] && rm -rf "$OUT/pred_frac_${TAG}/${CELL}"
  return $RC
}

for FR in 0.25 0.50 1.00; do
  for SEED in 0 1 2; do
    for MODE in r8 r2; do
      N=2; [ "$MODE" = "r2" ] && N=8
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
      one "$MODE" "$N" "$SEED" "$FR" &
    done
  done
done
wait
touch "$OUT/DONE_frac"
echo "=== $(date -u) FRAC sweep complete ===" >> "$MAIN"
