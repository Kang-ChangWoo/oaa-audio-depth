#!/bin/bash
# E144 finisher: shell-waits for all nine scored runs, then runs the pre-registered aggregation and
# appends its output to REPORT_E144.md. Pure shell `until` loops -- the LLM does not poll.
# The verdict prose is written by an agent afterwards, off the E144_ALL_DONE flag.
set -u
CODE=/root/storage/e144_code
COLLECT=/root/storage/e143_code/collect_e144
OUT=/root/local1/changwoo/e144
PY=${E144_PY:-/root/storage/envs/shared_audio/bin/python}
RUNS="ctrl_s0 ctrl_s1 ctrl_s2 bin4_s0 bin4_s1 bin4_s2 invfreq_s0 invfreq_s1 invfreq_s2"
mkdir -p $OUT/logs $OUT/results
L=$OUT/logs/finish.log
echo "$(date) finisher up, waiting for 9 eval JSONs in $COLLECT" >> $L
for r in $RUNS; do
  until [ -f $COLLECT/$r.json ]; do sleep 120; done
  echo "$(date) have $r.json" >> $L
done
for r in $RUNS; do
  if [ ! -f $COLLECT/$r.train_done.json ]; then
    echo "$(date) WARN $r.train_done.json missing (convergence table will be partial)" >> $L
  fi
done
# metric code md5, recorded again after the last run so "unmodified" is provable end to end
md5sum /root/storage/e143_code/e114_eval.py >> $L
E144_COLLECT=$COLLECT E144_AGG_OUT=$OUT/results/e144_agg.json \
  $PY $CODE/aggregate_e144.py > $OUT/results/e144_agg.txt 2>&1
RC=$?
echo "$(date) aggregate exit=$RC" >> $L
{
  echo
  echo "---"
  echo
  echo "## 13. 기계 집계 출력 — \`aggregate_e144.py\`, 사전등록 상수 그대로 (사후 수정 없음)"
  echo
  echo '```'
  cat $OUT/results/e144_agg.txt
  echo '```'
} >> $OUT/REPORT_E144.md
touch $OUT/E144_ALL_DONE
echo "$(date) finisher done, wrote E144_ALL_DONE" >> $L
