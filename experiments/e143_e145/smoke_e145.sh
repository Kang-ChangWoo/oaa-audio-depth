#!/bin/bash
# E145 smoke: prove the two-checkpoint mechanism BEFORE the nine 80-epoch runs (PREREG_E145 s4).
# Writes into a SEPARATE out dir so `--resume auto` can never pick a smoke last.pth up for a real run.
#   smk_ctrl     --loss-weight none    -> val_obj_m must equal val_mae_m every epoch,
#                                         best_own.pth must be BYTE-IDENTICAL to best.pth
#   smk_invfreq  --loss-weight invfreq -> val_obj_m must DIFFER from val_mae_m every epoch,
#                                         best_own.pth must exist and be tracked separately
# usage: smoke_e145.sh <gpu_ctrl> <gpu_invfreq> [epochs]
set -o pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export REPLICA_ROOT=/root/storage/replica_0422
export DATA_MODULE=data_0422_bin
export PYTHONPATH=/root/storage/e145_code:/root/storage/e143_code:/root/storage/implementation/shared_audio/hear360
CODE=/root/storage/e145_code
OUT=/root/local1/changwoo/e145/smoke
PY=${E145_PY:-/root/storage/envs/shared_audio/bin/python}
GA=${1:-0}
GB=${2:-1}
EP=${3:-4}
mkdir -p $OUT/out $OUT/logs
cd $OUT
echo "$(date) smoke start epochs=$EP gpus=$GA,$GB py=$PY" >> $OUT/logs/smoke.log
md5sum $CODE/train_oaa_e145.py >> $OUT/logs/smoke.log

E135_CUE=none CUDA_VISIBLE_DEVICES=$GA $PY $CODE/train_oaa_e145.py \
  --run-name smk_ctrl --cue none --loss-weight none \
  --nviews 2 --data-mode r2 --lr 5e-4 --warmup-ep 4 --epochs $EP \
  --batch-size 8 --accum 3 --seed 0 --out-dir $OUT/out >> $OUT/logs/smk_ctrl.log 2>&1 &
P1=$!
E135_CUE=none CUDA_VISIBLE_DEVICES=$GB $PY $CODE/train_oaa_e145.py \
  --run-name smk_invfreq --cue none --loss-weight invfreq \
  --nviews 2 --data-mode r2 --lr 5e-4 --warmup-ep 4 --epochs $EP \
  --batch-size 8 --accum 3 --seed 0 --out-dir $OUT/out >> $OUT/logs/smk_invfreq.log 2>&1 &
P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
echo "$(date) smoke exits ctrl=$R1 invfreq=$R2" >> $OUT/logs/smoke.log

$PY $CODE/check_smoke_e145.py $OUT/out/smk_ctrl $OUT/out/smk_invfreq 2>&1 | tee -a $OUT/logs/smoke.log
RC=${PIPESTATUS[0]}
if [ $RC -eq 0 ]; then touch $OUT/SMOKE_OK; else touch $OUT/SMOKE_FAIL; fi
echo "$(date) smoke check exit=$RC" >> $OUT/logs/smoke.log
exit $RC
