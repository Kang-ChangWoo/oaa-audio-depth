#!/bin/bash
# E146 status probe. Runs INSIDE container changwoo_audio_<node>. Invoked by PATH, never as an
# inline quoted string -- PREREG_E146 section 11 (E144's watcher died on triply-nested quoting).
#
#   ssh n1 docker exec changwoo_audio_n1 bash /root/storage/e146_code/status_e146.sh
#
# stderr is NOT redirected anywhere: a broken probe must be loud.
# Output contract, fixed field counts, whitespace separated:
#   E146STAT <node> <run> <alive|dead> <ep_done> <best_ep_mae> <best_ep_obj> <cells> <own_ckpt>   = 9 fields
#   E146SUM  <node> <n_runs> <n_alive> <n_traindone> <n_done> <n_fail>                            = 7 fields
OUT=/root/local1/changwoo/e146
# $1 = node label from the caller (the container's own hostname is an opaque container id)
NODE=${1:-$(hostname)}
RUNS=$(ls -1 $OUT/out 2>/dev/null)
NR=0; NA=0; NT=0; ND=0; NF=0
for RUN in $RUNS; do
  NR=$((NR + 1))
  if pgrep -f "train_oaa_e146.py --run-name $RUN " > /dev/null; then A=alive; NA=$((NA + 1)); else A=dead; fi
  EP=$(grep -c '^\[ep ' $OUT/logs/${RUN}.log 2>/dev/null); EP=${EP:-0}
  BM=-1; BO=-1
  if [ -f $OUT/out/${RUN}/train_done.json ]; then
    NT=$((NT + 1))
    BM=$(sed -n 's/.*"best_ep_mae":[ ]*\([0-9-]*\).*/\1/p' $OUT/out/${RUN}/train_done.json | head -1)
    BO=$(sed -n 's/.*"best_ep_obj":[ ]*\([0-9-]*\).*/\1/p' $OUT/out/${RUN}/train_done.json | head -1)
    BM=${BM:--1}; BO=${BO:--1}
  fi
  C=0
  [ -f $OUT/eval_results/${RUN}__mae.json ] && C=$((C + 1))
  [ -f $OUT/eval_results/${RUN}__own.json ] && C=$((C + 1))
  if [ -f $OUT/out/${RUN}/best_own.pth ]; then O=1; else O=0; fi
  [ -f $OUT/DONE_${RUN} ] && ND=$((ND + 1))
  [ -f $OUT/FAIL_${RUN} ] && NF=$((NF + 1))
  echo "E146STAT $NODE $RUN $A $EP $BM $BO $C $O"
done
echo "E146SUM $NODE $NR $NA $NT $ND $NF"
