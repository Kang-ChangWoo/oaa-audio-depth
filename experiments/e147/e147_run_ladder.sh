#!/usr/bin/env bash
# E147: extend the E142-fix fixed-reference grid to N = 24 and 32.  CPU ONLY, no GPU.
#
# This is a NEW FILE.  e142fix_run_all.sh / e142fix_cell.sh / e142_make_subset.py /
# e142_eval_fixedref.py / e114_eval.py are all reused UNMODIFIED; e142fix_cell.sh skips
# already-scored cells, so only the new N=24 / N=32 cells actually run.
set -uo pipefail

FIXCODE=/root/storage/e142fix_code
OUT=/root/local1/changwoo/e142fix          # same store, so existing cells are seen and skipped
E147=/root/local1/changwoo/e147
JOBS=${JOBS:-8}

mkdir -p "$OUT/eval_results" "$OUT/logs" "$E147/logs" "$E147/results"
MAIN="$E147/logs/run_ladder.log"
echo "=== $(date -u) E147 ladder start (JOBS=$JOBS) ===" >> "$MAIN"

echo "-- md5 BEFORE (must be 5e28675599c7872e34258a1038f503c8) --" >> "$MAIN"
md5sum /root/storage/e142_code/e114_eval.py \
       /root/storage/e142_code/e142_make_subset.py \
       "$FIXCODE/e142_eval_fixedref.py" >> "$MAIN"

# primary axis r2, secondary axis r8 (prereg sec 3)
for N in 24 32; do
  for SEED in 0 1 2 sp; do
    for MODE in r2 r8; do
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
      bash "$FIXCODE/e142fix_cell.sh" "$MODE" "$N" "$SEED" &
    done
  done
done
wait
echo "$(date -u) N=24/32 cells done" >> "$MAIN"

echo "-- md5 AFTER --" >> "$MAIN"
md5sum /root/storage/e142_code/e114_eval.py \
       /root/storage/e142_code/e142_make_subset.py \
       "$FIXCODE/e142_eval_fixedref.py" >> "$MAIN"

touch "$E147/DONE_ladder"
echo "=== $(date -u) E147 ladder complete ===" >> "$MAIN"
