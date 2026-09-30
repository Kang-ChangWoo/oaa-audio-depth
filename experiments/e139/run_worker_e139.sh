#!/bin/bash
# E139 worker = run_worker_e144.sh with (1) OUT -> e139, (2) EPOCHS default 40 (E144 ctrl's budget,
# PREREG sec.3), (3) e139_code FIRST on PYTHONPATH so `model` resolves to the patched oaa.py, and
# (4) the trainer is train_oaa_e139.py (a byte copy of train_oaa_e144.py -- the model, not the
# trainer, is what E139 changes). Log is appended, never truncated.
# usage: run_worker_e139.sh <gpu> "run:cue:lossweight:seed;..."
set -o pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export REPLICA_ROOT=/root/storage/replica_0422
export DATA_MODULE=data_0422_bin
export E139_K=${E139_K:-4}
# e139_code first (patched model + trainer), then e144/e143_code (args, data_0422_bin), then the repo.
export PYTHONPATH=/root/storage/e139_code:/root/storage/e144_code:/root/storage/e143_code:/root/storage/implementation/shared_audio/hear360
CODE=/root/storage/e139_code
OUT=/root/local1/changwoo/e139
GPU=$1
JOBS=$2
EPOCHS=${EPOCHS:-40}
# n4's container has no local /opt/conda; shared_audio is on NFS. Without this the worker exits 127
# with an empty log (the failure mode that cost E143 half a day).
PY=${E139_PY:-/root/storage/envs/shared_audio/bin/python}
mkdir -p $OUT/out $OUT/logs
cd $OUT
echo "$(date) GPU$GPU worker start: py=$PY epochs=$EPOCHS K=$E139_K trainer=$CODE/train_oaa_e139.py" >> logs/queue_gpu${GPU}.log
md5sum $CODE/train_oaa_e139.py $CODE/model/oaa.py >> logs/queue_gpu${GPU}.log 2>&1
IFS=";" read -ra JOBLIST <<< "$JOBS"
for job in "${JOBLIST[@]}"; do
  IFS=":" read -r RUN CUE LW SEED <<< "$job"
  while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
    if [ "$USED" -lt 500 ]; then break; fi
    echo "$(date) GPU$GPU busy (${USED}MiB), waiting..." >> logs/queue_gpu${GPU}.log
    sleep 60
  done
  echo "$(date) GPU$GPU starting $RUN (cue=$CUE lw=$LW seed=$SEED epochs=$EPOCHS K=$E139_K)" >> logs/queue_gpu${GPU}.log
  E135_CUE=$CUE CUDA_VISIBLE_DEVICES=$GPU $PY $CODE/train_oaa_e139.py \
    --run-name $RUN --cue $CUE --loss-weight $LW \
    --nviews 2 --data-mode r2 --lr 5e-4 --warmup-ep 4 --epochs $EPOCHS \
    --batch-size 8 --accum 3 --seed $SEED --resume auto \
    --out-dir $OUT/out >> logs/${RUN}.log 2>&1
  RC=$?
  echo "$(date) GPU$GPU finished $RUN exit=$RC" >> logs/queue_gpu${GPU}.log
  if [ "$RC" -eq 0 ]; then echo "$(date) rc=0" > $OUT/DONE_${RUN}; fi
done
echo "$(date) GPU$GPU queue done" >> logs/queue_gpu${GPU}.log
