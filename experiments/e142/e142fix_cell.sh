#!/usr/bin/env bash
# E142-fix: one cell = budget subset (unmodified e142_make_subset.py) -> e142_eval_fixedref.py
# Usage: e142fix_cell.sh <mode> <nviews> <seed>
set -uo pipefail

CODE=/root/storage/e142fix_code
SUBSET=/root/storage/e142_code/e142_make_subset.py     # reused as-is, selection is mode-independent
OUT=/root/local1/changwoo/e142fix
REF=$OUT/ref_fixed
PY=${E142_PY:-/opt/conda/bin/python}

MODE=$1; N=$2; SEED=$3
CELL="${MODE}_N${N}_s${SEED}"

mkdir -p "$OUT/eval_results" "$OUT/logs" "$OUT/pred_e142fix"
LOG="$OUT/logs/${CELL}.log"

echo "$(date -u) [start] $CELL" >> "$LOG"

if [ -f "$OUT/eval_results/${CELL}.json" ]; then
  echo "$(date -u) [skip] $CELL already scored" >> "$LOG"
  exit 0
fi

export PYTHONPATH=/root/storage/implementation/shared_audio/EchoRecon/src
export PYTHONDONTWRITEBYTECODE=1

"$PY" "$SUBSET" --mode "$MODE" --nviews "$N" --seed "$SEED" \
      --out-root "$OUT/pred_e142fix" >> "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then echo "$(date -u) [FAIL subset] $CELL rc=$RC" >> "$LOG"; exit 1; fi

"$PY" "$CODE/e142_eval_fixedref.py" --pred-dir "$OUT/pred_e142fix/${CELL}" \
      --ref-root "$REF" --out "$OUT/eval_results/${CELL}.json" >> "$LOG" 2>&1
RC=$?
echo "$(date -u) [done] $CELL rc=$RC" >> "$LOG"

if [ $RC -eq 0 ]; then
  rm -rf "$OUT/pred_e142fix/${CELL}"     # reproducible from the recorded .select.json
fi
exit $RC
