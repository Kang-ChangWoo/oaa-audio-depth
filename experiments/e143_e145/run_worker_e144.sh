#!/bin/bash
# E144 worker: run_worker_e143.sh with (1) OUT -> e144, (2) EPOCHS default 80, (3) the training log
# APPENDED not truncated (`>` is what permanently lost E143's training logs, and an 80-epoch run is
# ~4.5 h of output to lose), (4) the trainer is train_oaa_e144.py (per-epoch atomic last.pth) and
# (5) --resume auto, so re-running this worker after a crash continues instead of restarting.
# usage: run_worker_e144.sh <gpu> "run:cue:lossweight:seed;..."
set -o pipefail
# Mixed-GPU nodes (n6 = 7x 3090 + 1x A100): without this, CUDA orders devices fastest-first, so
# CUDA_VISIBLE_DEVICES=N does not mean nvidia-smi index N.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export REPLICA_ROOT=/root/storage/replica_0422
export DATA_MODULE=data_0422_bin
# e144_code first (trainer), then e143_code (data_0422_bin, gate0_bin_dist.json), then the model.
export PYTHONPATH=/root/storage/e144_code:/root/storage/e143_code:/root/storage/implementation/shared_audio/hear360
CODE=/root/storage/e144_code
E143CODE=/root/storage/e143_code
OUT=/root/local1/changwoo/e144
GPU=$1
JOBS=$2
EPOCHS=${EPOCHS:-80}
# n4's container has no local /opt/conda -- shared_audio lives on NFS there. Without an explicit
# path the worker exits 127 with an empty log.
PY=${E144_PY:-/opt/conda/envs/shared_audio/bin/python}
mkdir -p $OUT/out $OUT/logs
cd $OUT
echo "$(date) GPU$GPU worker start: py=$PY epochs=$EPOCHS trainer=$CODE/train_oaa_e144.py" >> logs/queue_gpu${GPU}.log
md5sum $CODE/train_oaa_e144.py $E143CODE/e114_eval.py >> logs/queue_gpu${GPU}.log 2>&1
IFS=";" read -ra JOBLIST <<< "$JOBS"
for job in "${JOBLIST[@]}"; do
  IFS=":" read -r RUN CUE LW SEED <<< "$job"
  while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
    if [ "$USED" -lt 500 ]; then break; fi
    echo "$(date) GPU$GPU busy (${USED}MiB), waiting..." >> logs/queue_gpu${GPU}.log
    sleep 60
  done
  echo "$(date) GPU$GPU starting $RUN (cue=$CUE lw=$LW seed=$SEED epochs=$EPOCHS)" >> logs/queue_gpu${GPU}.log
  E135_CUE=$CUE CUDA_VISIBLE_DEVICES=$GPU $PY $CODE/train_oaa_e144.py \
    --run-name $RUN --cue $CUE --loss-weight $LW \
    --nviews 2 --data-mode r2 --lr 5e-4 --warmup-ep 4 --epochs $EPOCHS \
    --batch-size 8 --accum 3 --seed $SEED --resume auto \
    --out-dir $OUT/out >> logs/${RUN}.log 2>&1
  RC=$?
  echo "$(date) GPU$GPU finished $RUN exit=$RC" >> logs/queue_gpu${GPU}.log
done
echo "$(date) GPU$GPU queue done" >> logs/queue_gpu${GPU}.log
