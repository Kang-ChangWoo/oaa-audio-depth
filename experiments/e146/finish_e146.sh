#!/bin/bash
# E146 finisher: ONE window on n4. Shell-waits (no LLM polling) for all 12 eval cells to land in the
# shared NFS collect dir, runs aggregate_e146.py (PREREG constants hardcoded), appends the verdict to
# REPORT_E146.md, then touches E146_ALL_DONE for the watcher automation.
set -o pipefail
CODE=/root/storage/e146_code
OUT=/root/local1/changwoo/e146
PY=${E146_PY:-/root/storage/envs/shared_audio/bin/python}
LOG=$OUT/logs/finish.log
mkdir -p $OUT/logs $OUT/results
RUNS="ctrl_s0 ctrl_s1 ctrl_s2 invfreq_s0 invfreq_s1 invfreq_s2"

echo "$(date) finisher start; waiting for 12 cells in $CODE/collect_e146" >> $LOG
while true; do
  N=0
  for R in $RUNS; do
    for T in mae own; do
      [ -f $CODE/collect_e146/${R}__${T}.json ] && N=$((N + 1))
    done
  done
  if [ "$N" -ge 12 ]; then break; fi
  echo "$(date) cells=$N/12" >> $LOG
  sleep 120
done
echo "$(date) all 12 cells present; aggregating" >> $LOG

$PY $CODE/aggregate_e146.py $OUT/results/e146_agg.json > $OUT/logs/aggregate.log 2>&1
RC=$?
echo "$(date) aggregate exit=$RC" >> $LOG
{
  echo
  echo "## 13. Auto-aggregated verdict ($(date))"
  echo
  echo '```'
  cat $OUT/logs/aggregate.log
  echo '```'
} >> $OUT/REPORT_E146.md
touch $OUT/E146_ALL_DONE
echo "$(date) finisher done, E146_ALL_DONE touched" >> $LOG
