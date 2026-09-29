#!/usr/bin/env bash
# E142 full grid: mode {r2,r8} x N {1,2,4,8,16} x seed {0,1,2,sp}  = 40 cells, CPU only.
# Plus mode-independent geometry/reference diagnostics (PREREG 2-5).
set -uo pipefail

CODE=/root/storage/e142_code
OUT=/root/local1/changwoo/e142
PY=${E142_PY:-/opt/conda/bin/python}
JOBS=${JOBS:-8}

mkdir -p "$OUT/eval_results" "$OUT/logs" "$OUT/diag" "$OUT/results"
MAIN="$OUT/logs/run_all.log"
echo "=== $(date -u) E142 grid start (JOBS=$JOBS) ===" >> "$MAIN"

run_cell() { bash "$CODE/e142_cell.sh" "$1" "$2" "$3"; }

for N in 1 2 4 8 16; do
  for SEED in 0 1 2 sp; do
    for MODE in r2 r8; do
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
      run_cell "$MODE" "$N" "$SEED" &
    done
  done
done
wait
echo "$(date -u) all eval cells done" >> "$MAIN"

# --- diagnostics: selections are mode-independent, so compute once per (N,seed) via the r2 cell.
export PYTHONPATH=/root/storage/implementation/shared_audio/EchoRecon/src
export PYTHONDONTWRITEBYTECODE=1
for N in 1 2 4 8 16; do
  for SEED in 0 1 2 sp; do
    SJ="$OUT/pred_e142/r2_N${N}_s${SEED}.select.json"
    [ -f "$SJ" ] || { echo "$(date -u) [diag skip] no $SJ" >> "$MAIN"; continue; }
    # reference-cloud size only for seed 0 and sp (it is the expensive part and N-driven)
    EXTRA="--extent-only"
    if [ "$SEED" = "0" ] || [ "$SEED" = "sp" ]; then EXTRA=""; fi
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    ( "$PY" "$CODE/e142_diag.py" --select-json "$SJ" \
        --out "$OUT/diag/N${N}_s${SEED}.json" $EXTRA \
        >> "$OUT/logs/diag_N${N}_s${SEED}.log" 2>&1 ) &
  done
done
wait
echo "$(date -u) all diag done" >> "$MAIN"

touch "$OUT/DONE_e142"
echo "=== $(date -u) E142 grid complete ===" >> "$MAIN"
