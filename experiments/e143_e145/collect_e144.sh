#!/bin/bash
# E144 per-node collector: shell-waits for each local run's train_done.json and copies it to the
# shared NFS collect dir, so one aggregate call sees the convergence history of runs from both
# nodes. Shell `until` loop -- no LLM polling. usage: collect_e144.sh <run> ...
set -u
OUT=/root/local1/changwoo/e144
DEST=/root/storage/e143_code/collect_e144
mkdir -p $DEST $OUT/logs
for RUN in "$@"; do
  echo "$(date) waiting for $RUN" >> $OUT/logs/collect.log
  until [ -f $OUT/out/$RUN/train_done.json ]; do sleep 60; done
  cp $OUT/out/$RUN/train_done.json $DEST/${RUN}.train_done.json
  echo "$(date) collected $RUN -> $DEST/${RUN}.train_done.json" >> $OUT/logs/collect.log
done
echo "$(date) collector done: $*" >> $OUT/logs/collect.log
