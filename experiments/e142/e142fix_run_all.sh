#!/usr/bin/env bash
# E142-fix full grid: mode {r2,r8} x N {1,2,4,8,16} x seed {0,1,2,sp} = 40 cells
#                     + N=0 (all steps) gate x 2 modes = 42 cells.  CPU only, no GPU.
set -uo pipefail

CODE=/root/storage/e142fix_code
OUT=/root/local1/changwoo/e142fix
PY=${E142_PY:-/opt/conda/bin/python}
JOBS=${JOBS:-8}

mkdir -p "$OUT/eval_results" "$OUT/logs" "$OUT/results"
MAIN="$OUT/logs/run_all.log"
echo "=== $(date -u) E142fix grid start (JOBS=$JOBS) ===" >> "$MAIN"

# md5 of the untouched original, recorded before the grid
md5sum /root/storage/e142_code/e114_eval.py >> "$MAIN"

for N in 1 2 4 8 16; do
  for SEED in 0 1 2 sp; do
    for MODE in r2 r8; do
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
      bash "$CODE/e142fix_cell.sh" "$MODE" "$N" "$SEED" &
    done
  done
done
wait
echo "$(date -u) 40 budget cells done" >> "$MAIN"

# self-consistency gate: N=0 means every step, so the fixed reference IS the selected reference
# and this must reproduce e114_eval.py's N=all numbers exactly.
for MODE in r2 r8; do
  bash "$CODE/e142fix_cell.sh" "$MODE" 0 0 &
done
wait
echo "$(date -u) N=all gate cells done" >> "$MAIN"

md5sum /root/storage/e142_code/e114_eval.py >> "$MAIN"

touch "$OUT/DONE_e142fix"
echo "=== $(date -u) E142fix grid complete ===" >> "$MAIN"
