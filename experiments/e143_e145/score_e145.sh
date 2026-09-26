#!/bin/bash
# E145 scorer, one per run: shell-polls for its own run, then predicts + evaluates BOTH checkpoints.
# No LLM polling -- `until [ -f ... ]; do sleep 60; done` (PREREG_E145 section 11).
#
#   best.pth      selected on val UNWEIGHTED MAE   -> reproduces E144, gate G1
#   best_own.pth  selected on val ARM-WEIGHTED MAE -> the arm's own objective
#
# Metric code is e114_eval.py UNMODIFIED (md5 5e28675599c7872e34258a1038f503c8), logged on both
# sides of every eval, PREREG_E145 section 9.
#
# Integrity check (PREREG_E145 section 4): for --loss-weight none the two checkpoints must hold
# identical CONTENT. Not file bytes -- torch.save names the zip container after the output filename
# (best.pth/data.pkl vs best_own.pth/data.pkl), so identical weights never give identical files;
# measured on the E145 smoke. ckpt_content_hash.py hashes state_dict + args instead.
# When the content matches, the `own` cell is filled from that proven identity instead of a second
# prediction pass -- EXCEPT for the run named in FORCE_BOTH, scored twice end-to-end to show the
# eval path is deterministic.
# usage: score_e145.sh <run_name> <gpu>
set -o pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CODE=/root/storage/e143_code
E145CODE=/root/storage/e145_code
OUT=/root/local1/changwoo/e145
PY=${E145_PY:-/root/storage/envs/shared_audio/bin/python}
RUN=$1
GPU=$2
FORCE_BOTH=${FORCE_BOTH:-ctrl_s0}
LOG=$OUT/logs/score_${RUN}.log
mkdir -p $OUT/eval_results $OUT/logs $OUT/results $OUT/pred $E145CODE/collect_e145
cd $OUT

echo "$(date) waiting for $RUN" >> $LOG
until [ -f out/${RUN}/train_done.json ]; do sleep 60; done

CK_MAE=$OUT/out/${RUN}/best.pth
CK_OWN=$OUT/out/${RUN}/best_own.pth
for f in $CK_MAE $CK_OWN; do
  if [ ! -f $f ]; then
    echo "$(date) ABORT $RUN: train_done but missing $f" >> $LOG
    touch $OUT/FAIL_${RUN}
    exit 1
  fi
done

# ---- PREREG section 4 integrity check, recorded whatever the arm ----
H_MAE=$($PY $E145CODE/ckpt_content_hash.py $CK_MAE | cut -d" " -f1)
H_OWN=$($PY $E145CODE/ckpt_content_hash.py $CK_OWN | cut -d" " -f1)
if [ -z "$H_MAE" ] || [ -z "$H_OWN" ]; then
  echo "$(date) ABORT $RUN: content hash failed" >> $LOG; touch $OUT/FAIL_${RUN}; exit 1
fi
if [ "$H_MAE" = "$H_OWN" ]; then IDENT=IDENTICAL; else IDENT=DIFFERENT; fi
echo "$(date) $RUN ckpt_content_hash best=$H_MAE best_own=$H_OWN -> $IDENT" >> $LOG
echo "{\"run\":\"$RUN\",\"content_best\":\"$H_MAE\",\"content_best_own\":\"$H_OWN\",\"identical\":\"$IDENT\"}" \
  > $OUT/results/integrity_${RUN}.json
cp $OUT/results/integrity_${RUN}.json $E145CODE/collect_e145/integrity_${RUN}.json
# runs are spread over two nodes; the aggregator reads best_ep under both criteria from here
cp $OUT/out/${RUN}/train_done.json $E145CODE/collect_e145/traindone_${RUN}.json

export DATA_MODULE=data_0422
export REPLICA_ROOT=/root/storage/replica_0422
export PYTHONPATH=$CODE:/root/storage/implementation/shared_audio/hear360

# predict + eval one checkpoint into its own tagged dirs
score_one () {
  local TAG=$1 CKPT=$2
  md5sum $CODE/e114_eval.py >> $LOG          # before
  echo "$(date) predict $RUN[$TAG] on GPU$GPU ckpt=$CKPT" >> $LOG
  for sc in apartment_2 frl_apartment_5 office_4; do
    SEQS=$($PY $CODE/list_seqs.py $sc) || return 1
    for sq in $SEQS; do
      $PY $CODE/e143_predict.py --scene $sc --seq $sq --ckpt $CKPT --gpu $GPU \
          --out-dir $OUT/pred/${RUN}__${TAG} >> $OUT/logs/predict_${RUN}__${TAG}.log 2>&1 || return 1
    done
  done
  echo "$(date) predict done $RUN[$TAG]" >> $LOG
  $PY $CODE/e114_eval.py --pred-dir $OUT/pred/${RUN}__${TAG}/r2 \
      --out $OUT/eval_results/${RUN}__${TAG}.json > $OUT/logs/eval_${RUN}__${TAG}.log 2>&1
  local RC=$?
  md5sum $CODE/e114_eval.py >> $LOG          # after
  echo "$(date) eval done $RUN[$TAG] exit=$RC" >> $LOG
  [ $RC -eq 0 ] && cp $OUT/eval_results/${RUN}__${TAG}.json $E145CODE/collect_e145/${RUN}__${TAG}.json
  return $RC
}

score_one mae $CK_MAE
RC1=$?

if [ "$IDENT" = "IDENTICAL" ] && [ "$RUN" != "$FORCE_BOTH" ]; then
  # Byte-identical checkpoint: a second prediction pass would be scoring the same weights again.
  # Fill the cell from the proven identity and say so in the JSON.
  echo "$(date) $RUN[own] filled from CONTENT identity with best.pth (no second pass)" >> $LOG
  if [ $RC1 -eq 0 ]; then
    $PY - "$OUT/eval_results/${RUN}__mae.json" "$OUT/eval_results/${RUN}__own.json" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
j = json.load(open(src))
j["e145_note"] = "filled from content-identical best.pth (PREREG_E145 s4 integrity); not a second eval"
json.dump(j, open(dst, "w"), indent=1)
PYEOF
    cp $OUT/eval_results/${RUN}__own.json $E145CODE/collect_e145/${RUN}__own.json
  fi
  RC2=$RC1
else
  score_one own $CK_OWN
  RC2=$?
fi

echo "$(date) score_e145 done $RUN mae_exit=$RC1 own_exit=$RC2 ident=$IDENT" >> $LOG
if [ $RC1 -eq 0 ] && [ $RC2 -eq 0 ]; then touch $OUT/DONE_${RUN}; else touch $OUT/FAIL_${RUN}; fi
