#!/usr/bin/env bash
# E147b: re-score the two E147 position ladders at --frac 0.50 and 1.00 (plus a 0.25
# inertness re-run for gate G2).  CPU ONLY, no GPU.
#
# NEW FILE.  e142_make_subset.py / e142_eval_fixedref.py / e114_eval.py are reused
# UNMODIFIED.  Every frac goes to its own output dir so the registered FRAC=0.25 grid
# in eval_results/ is never written to.
set -uo pipefail

CODE=/root/storage/e142fix_code
SUBSET=/root/storage/e142_code/e142_make_subset.py
OUT=/root/local1/changwoo/e142fix
REF=$OUT/ref_fixed
E147B=/root/local1/changwoo/e147b
PY=${E142_PY:-/opt/conda/bin/python}
JOBS=${JOBS:-8}

export PYTHONPATH=/root/storage/implementation/shared_audio/EchoRecon/src
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$E147B/logs" "$E147B/results" "$OUT/e147b_frac"
MAIN="$E147B/logs/run_frac.log"
echo "=== $(date -u) E147b frac ladders start (JOBS=$JOBS) ===" >> "$MAIN"

echo "-- md5 BEFORE (e114_eval.py must be 5e28675599c7872e34258a1038f503c8) --" >> "$MAIN"
md5sum /root/storage/e142_code/e114_eval.py \
       /root/storage/e142_code/e142_make_subset.py \
       "$CODE/e142_eval_fixedref.py" >> "$MAIN"

one() {  # mode N seed frac
  local MODE=$1 N=$2 SEED=$3 FR=$4
  local TAG="f${FR}"
  local CELL="${MODE}_N${N}_s${SEED}"
  local DEST="$OUT/e147b_frac/${TAG}"
  local LOG="$E147B/logs/${TAG}_${CELL}.log"
  mkdir -p "$DEST"
  if [ -f "$DEST/${CELL}.json" ]; then
    echo "$(date -u) [skip] $TAG $CELL already scored" >> "$LOG"; return 0
  fi
  "$PY" "$SUBSET" --mode "$MODE" --nviews "$N" --seed "$SEED" \
        --out-root "$OUT/pred_e147b_${TAG}" >> "$LOG" 2>&1 || {
        echo "$(date -u) [FAIL subset] $TAG $CELL" >> "$LOG"; return 1; }
  "$PY" "$CODE/e142_eval_fixedref.py" --pred-dir "$OUT/pred_e147b_${TAG}/${CELL}" \
        --ref-root "$REF" --frac "$FR" --out "$DEST/${CELL}.json" >> "$LOG" 2>&1
  local RC=$?
  echo "$(date -u) [done] $TAG $CELL rc=$RC" >> "$LOG"
  [ $RC -eq 0 ] && rm -rf "$OUT/pred_e147b_${TAG}/${CELL}"
  return $RC
}

# union of the two ladders' N, r2 only, seeds 0/1/2.
# 0.50 = the registered comparison arm; 1.00 = trend arm; 0.25 = inertness gate G2.
for FR in 0.50 1.00 0.25; do
  for N in 1 2 4 8 16 24 32; do
    for SEED in 0 1 2; do
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
      one r2 "$N" "$SEED" "$FR" &
    done
  done
done
wait
echo "$(date -u) all frac cells done" >> "$MAIN"

echo "-- md5 AFTER --" >> "$MAIN"
md5sum /root/storage/e142_code/e114_eval.py \
       /root/storage/e142_code/e142_make_subset.py \
       "$CODE/e142_eval_fixedref.py" >> "$MAIN"

touch "$E147B/DONE_frac"
echo "=== $(date -u) E147b frac ladders complete ===" >> "$MAIN"
