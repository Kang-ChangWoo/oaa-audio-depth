#!/bin/bash
# E139 G0 gate + fan-out, run by the shell so no agent polls. PREREG sec.6:
#   G0 = ep00 val_MAE must be < 3x the E144 ctrl ep00 baseline (2.4464 m) -> 7.3392 m.
# Pass  -> launch seeds 1 and 2 on GPUs 5 and 6.
# Fail  -> kill seed 0, write GATE_G0_FAIL, launch nothing.
OUT=/root/local1/changwoo/e139
LOG=$OUT/logs/distq_s0.log
GATE=$OUT/logs/gate_g0.log
LIMIT=7.3392
echo "$(date) waiting for ep00 in $LOG (limit $LIMIT)" >> $GATE
until grep -q "^\[ep 00\]" $LOG 2>/dev/null; do
  if ! pgrep -f "train_oaa_e139.py --run-name distq_s0" >/dev/null; then
    echo "$(date) seed0 process gone before ep00 -- aborting fan-out" >> $GATE
    echo "process died before ep00" > $OUT/GATE_G0_FAIL
    exit 1
  fi
  sleep 30
done
MAE=$(grep -m1 "^\[ep 00\]" $LOG | sed -n 's/.*val_MAE=\([0-9.]*\)m.*/\1/p')
echo "$(date) ep00 val_MAE=$MAE limit=$LIMIT" >> $GATE
if [ -z "$MAE" ]; then
  echo "$(date) could not parse val_MAE -- treating as FAIL" >> $GATE
  echo "unparseable ep00 line" > $OUT/GATE_G0_FAIL
  exit 1
fi
if awk -v m="$MAE" -v l="$LIMIT" 'BEGIN{exit !(m+0 < l+0)}'; then
  echo "$(date) G0 PASS -> launching distq_s1 (GPU5), distq_s2 (GPU6)" >> $GATE
  echo "ep00 val_MAE=$MAE < $LIMIT" > $OUT/GATE_G0_PASS
  tmux new-window -d -t claude -n e139_s1 2>/dev/null
  tmux send-keys -t claude:e139_s1 "export E139_PY=/root/storage/envs/shared_audio/bin/python E139_K=4 EPOCHS=40; bash /root/storage/e139_code/run_worker_e139.sh 5 distq_s1:none:none:1" C-m
  tmux new-window -d -t claude -n e139_s2 2>/dev/null
  tmux send-keys -t claude:e139_s2 "export E139_PY=/root/storage/envs/shared_audio/bin/python E139_K=4 EPOCHS=40; bash /root/storage/e139_code/run_worker_e139.sh 6 distq_s2:none:none:2" C-m
else
  echo "$(date) G0 FAIL (val_MAE=$MAE >= $LIMIT) -> killing seed0, no fan-out" >> $GATE
  echo "ep00 val_MAE=$MAE >= $LIMIT" > $OUT/GATE_G0_FAIL
  pkill -f "train_oaa_e139.py --run-name distq_s0"
  exit 1
fi
# one watcher for the whole experiment: wait for all three DONE markers, then mark the block
echo "$(date) watcher: waiting for DONE_distq_s0/s1/s2" >> $GATE
until [ -f $OUT/DONE_distq_s0 ] && [ -f $OUT/DONE_distq_s1 ] && [ -f $OUT/DONE_distq_s2 ]; do
  sleep 300
done
echo "$(date) all three seeds done" >> $GATE
date > $OUT/E139_ALL_DONE
