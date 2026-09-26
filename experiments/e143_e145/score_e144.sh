#!/bin/bash
# E144 scorer, one per run: shell-polls for its own run, then predicts + evaluates it.
# No LLM polling -- `until [ -f ... ]; do sleep 60; done`.
# Metric code is e114_eval.py UNMODIFIED (md5 5e28675599c7872e34258a1038f503c8), PREREG_E144 sec. 6.
# usage: score_e144.sh <run_name> <gpu>
set -o pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CODE=/root/storage/e143_code
OUT=/root/local1/changwoo/e144
PY=${E144_PY:-/opt/conda/envs/shared_audio/bin/python}
RUN=$1
GPU=$2
mkdir -p $OUT/eval_results $OUT/logs $OUT/results $OUT/pred
cd $OUT

echo "$(date) waiting for $RUN" >> logs/score_${RUN}.log
until [ -f out/${RUN}/train_done.json ]; do sleep 60; done

CKPT=$OUT/out/${RUN}/best.pth
if [ ! -f $CKPT ]; then
  echo "$(date) ABORT $RUN: train_done but no best.pth" >> logs/score_${RUN}.log
  exit 1
fi

# md5 of the metric code, recorded on both sides of the run so "unmodified" is provable.
md5sum $CODE/e114_eval.py >> logs/score_${RUN}.log

export DATA_MODULE=data_0422
export REPLICA_ROOT=/root/storage/replica_0422
export PYTHONPATH=$CODE:/root/storage/implementation/shared_audio/hear360
echo "$(date) predict $RUN on GPU$GPU" >> logs/score_${RUN}.log
set -e
for sc in apartment_2 frl_apartment_5 office_4; do
  SEQS=$($PY $CODE/list_seqs.py $sc)
  for sq in $SEQS; do
    $PY $CODE/e143_predict.py --scene $sc --seq $sq --ckpt $CKPT --gpu $GPU \
        --out-dir $OUT/pred/${RUN} >> logs/predict_${RUN}.log 2>&1
  done
done
set +e
echo "$(date) predict done $RUN" >> logs/score_${RUN}.log

$PY $CODE/e114_eval.py --pred-dir $OUT/pred/${RUN}/r2 \
    --out $OUT/eval_results/${RUN}.json > logs/eval_${RUN}.log 2>&1
RC=$?
md5sum $CODE/e114_eval.py >> logs/score_${RUN}.log
echo "$(date) eval done $RUN exit=$RC" >> logs/score_${RUN}.log

# Runs are spread across nodes; collect the per-seed JSON onto the shared NFS dir so one
# aggregate call can see all nine runs.
mkdir -p $CODE/collect_e144
if [ $RC -eq 0 ]; then cp $OUT/eval_results/${RUN}.json $CODE/collect_e144/${RUN}.json; fi
touch $OUT/DONE_${RUN}
