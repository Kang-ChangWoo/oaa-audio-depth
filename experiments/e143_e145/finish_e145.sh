#!/bin/bash
# E145 finisher: ONE window on n4. Shell-waits (no LLM polling) for all 18 eval cells to land in the
# shared NFS collect dir, runs aggregate_e145.py (PREREG constants hardcoded), appends the verdict to
# REPORT_E145.md, then touches E145_ALL_DONE for the watcher automation.
set -o pipefail
CODE=/root/storage/e145_code
OUT=/root/local1/changwoo/e145
PY=${E145_PY:-/root/storage/envs/shared_audio/bin/python}
LOG=$OUT/logs/finish.log
mkdir -p $OUT/logs $OUT/results
RUNS="ctrl_s0 ctrl_s1 ctrl_s2 bin4_s0 bin4_s1 bin4_s2 invfreq_s0 invfreq_s1 invfreq_s2"

echo "$(date) finisher start; waiting for 18 cells in $CODE/collect_e145" >> $LOG
while true; do
  N=0
  for R in $RUNS; do
    for T in mae own; do
      [ -f $CODE/collect_e145/${R}__${T}.json ] && N=$((N + 1))
    done
  done
  if [ "$N" -ge 18 ]; then break; fi
  echo "$(date) cells=$N/18" >> $LOG
  sleep 120
done
echo "$(date) all 18 cells present; aggregating" >> $LOG

$PY $CODE/aggregate_e145.py $OUT/results/e145_agg.json > $OUT/logs/aggregate.log 2>&1
RC=$?
echo "$(date) aggregate exit=$RC" >> $LOG
{
  echo
  echo "## 13. Auto-aggregated verdict ($(date))"
  echo
  echo '```'
  cat $OUT/logs/aggregate.log
  echo '```'
} >> $OUT/REPORT_E145.md
touch $OUT/E145_ALL_DONE
echo "$(date) finisher done, E145_ALL_DONE touched" >> $LOG
