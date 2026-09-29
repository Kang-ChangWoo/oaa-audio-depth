#!/usr/bin/env bash
# E142: one cell = build budget subset -> e114_eval.py UNMODIFIED -> eval_results/<cell>.json
# Usage: e142_cell.sh <mode> <nviews> <seed>
set -uo pipefail

CODE=/root/storage/e142_code
OUT=/root/local1/changwoo/e142
PY=${E142_PY:-/opt/conda/bin/python}

MODE=$1; N=$2; SEED=$3
CELL="${MODE}_N${N}_s${SEED}"

mkdir -p "$OUT/eval_results" "$OUT/logs" "$OUT/pred_e142" "$OUT/diag"
LOG="$OUT/logs/${CELL}.log"

echo "$(date -u) [start] $CELL" >> "$LOG"

if [ -f "$OUT/eval_results/${CELL}.json" ]; then
  echo "$(date -u) [skip] $CELL already scored" >> "$LOG"
  exit 0
fi

export PYTHONPATH=/root/storage/implementation/shared_audio/EchoRecon/src
export PYTHONDONTWRITEBYTECODE=1

"$PY" "$CODE/e142_make_subset.py" --mode "$MODE" --nviews "$N" --seed "$SEED" \
      --out-root "$OUT/pred_e142" >> "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then echo "$(date -u) [FAIL subset] $CELL rc=$RC" >> "$LOG"; exit 1; fi

# e114_eval.py is used with no edits; the budget lives entirely in the npz we just wrote.
"$PY" "$CODE/e114_eval.py" --pred-dir "$OUT/pred_e142/${CELL}" \
      --out "$OUT/eval_results/${CELL}.json" >> "$LOG" 2>&1
RC=$?
echo "$(date -u) [done] $CELL rc=$RC" >> "$LOG"

if [ $RC -eq 0 ]; then
  # subset npz are large and fully reproducible from the recorded selection -- drop them.
  rm -rf "$OUT/pred_e142/${CELL}"
fi
exit $RC
