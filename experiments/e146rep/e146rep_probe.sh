#!/bin/bash
# E146REP probe. Runs INSIDE changwoo_audio_n4. Prints exactly one line:
#   probe alive=<n> traindone=<n> cells=<n> done=<n> fail=<csv|-> epoch=<n>
# A FILE so the watcher never nests shell quoting (the E144 watcher failure mode).
set -u
OUT=/root/local1/changwoo/e146rep
RUN=ctrl_s2_rep1

# DISTINCT runs alive: pgrep -fc would also count the 6 DataLoader workers sharing the cmdline.
ALIVE=$(ps -eo args 2>/dev/null | grep -o "train_oaa_e146\.py --run-name ctrl_s2_rep1" | sort -u | wc -l | tr -d " ")
case $ALIVE in ""|*[!0-9]*) ALIVE=0 ;; esac

TD=0; [ -f "$OUT/out/$RUN/train_done.json" ] && TD=1
DONE=0; [ -f "$OUT/DONE_$RUN" ] && DONE=1
FAILS="-"; [ -f "$OUT/FAIL_$RUN" ] && FAILS=$RUN

# 2026-09-28 fix: count the PRIMARY write of score_rep.sh score_one() -- $OUT/eval_results/ --
# not the cross-node convenience copy. The old line counted
# /root/storage/e146rep_code/collect_e146rep/, but score_rep.sh copies to
# $E146CODE=/root/storage/e146_code/collect_e146rep/ (one "rep" apart, inherited from
# score_e146.sh). cells stayed 0/2 for 5 h with done=1 and the watcher never fired.
# $OUT is the ONE path both scripts already agree on, so the count cannot drift again.
CELLS=0
for T in mae own; do
  [ -f "$OUT/eval_results/${RUN}__${T}.json" ] && CELLS=$((CELLS + 1))
done

# last logged epoch, so a stall is visible as a frozen counter rather than only as alive=0
EP=$(grep -o "^\[ep [0-9]*\]" "$OUT/logs/${RUN}.log" 2>/dev/null | tail -1 | grep -o "[0-9]*")
case ${EP:-} in ""|*[!0-9]*) EP=-1 ;; esac

echo "probe alive=$ALIVE traindone=$TD cells=$CELLS done=$DONE fail=$FAILS epoch=$EP"
