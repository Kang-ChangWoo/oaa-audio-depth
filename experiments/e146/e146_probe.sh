#!/bin/bash
# E146 probe. Runs INSIDE changwoo_audio_n4. Prints exactly one line:
#   probe alive=<n> traindone=<n> cells=<n> done=<n> fail=<csv|-> alldone=<0|1>
# Kept as a FILE so the watcher never has to nest shell quoting (the E144 watcher failure mode).
set -u
OUT=/root/local1/changwoo/e146
CODE=/root/storage/e146_code
RUNS="ctrl_s0 ctrl_s1 ctrl_s2 invfreq_s0 invfreq_s1 invfreq_s2"

# DISTINCT runs alive. pgrep -fc would count DataLoader workers too (num_workers=6 share the
# parent's cmdline), so 6 runs would report alive=42 and a half-dead queue would still look healthy.
ALIVE=$(ps -eo args 2>/dev/null | grep -o "train_oaa_e146\.py --run-name [A-Za-z0-9_]*" | sort -u | wc -l | tr -d " ")
case $ALIVE in ''|*[!0-9]*) ALIVE=0 ;; esac

TD=0
DONE=0
FAILS=""
for R in $RUNS; do
  [ -f "$OUT/out/$R/train_done.json" ] && TD=$((TD + 1))
  [ -f "$OUT/DONE_$R" ] && DONE=$((DONE + 1))
  [ -f "$OUT/FAIL_$R" ] && FAILS="$FAILS,$R"
done
FAILS=${FAILS#,}
[ -z "$FAILS" ] && FAILS="-"

CELLS=0
for R in $RUNS; do
  for T in mae own; do
    [ -f "$CODE/collect_e146/${R}__${T}.json" ] && CELLS=$((CELLS + 1))
  done
done

ALLDONE=0
[ -f "$OUT/E146_ALL_DONE" ] && ALLDONE=1

echo "probe alive=$ALIVE traindone=$TD cells=$CELLS done=$DONE fail=$FAILS alldone=$ALLDONE"
