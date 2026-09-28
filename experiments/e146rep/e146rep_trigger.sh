#!/bin/bash
# E146REP watcher trigger (ctrl_s2_rep1 repeat-noise probe, n4 GPU5).
# Runs ON THE MACMINI, reads n4 over ssh, prints ONE line and sets an exit code.
#
#   exit 0 = FIRE (wake the agent)      exit 1 = QUIET (keep looping)
# Schedule is `on-exit until <this script>; do sleep 900; done`, so a 0 stops the loop.
#
# Same three hard-won rules as e146_trigger.sh:
#   * the remote work lives in a FILE on the node (e146rep_probe.sh), invoked by name
#   * stderr is KEPT and surfaced in the output line, never discarded
#   * an unreadable probe is an EXPLICIT state, and three in a row FIRES instead of staying silent
#
# Output: one line.
#   E146REP state=<STATE> alive=<n> traindone=<n> cells=<n>/2 done=<n> fail=<csv|-> epoch=<n> note=<text>
# STATE is one of: RUNNING DONE FAILED STALLED UNREADABLE
set -u
NODE=${E146REP_NODE:-n4}
CTR=${E146REP_CTR:-changwoo_audio_n4}
# overridable so the STATE MACHINE can be exercised against a stub probe without waiting for a run
PROBE=${E146REP_PROBE:-/root/storage/e146rep_code/e146rep_probe.sh}
STATEDIR=${E146REP_STATEDIR:-$HOME/.openclaw/workspace/state}
UNREAD_FILE=$STATEDIR/e146rep_unreadable_count
STALL_FILE=$STATEDIR/e146rep_stall_count
INCON_FILE=$STATEDIR/e146rep_inconsistent_count
mkdir -p "$STATEDIR"

num () { case ${1:-} in ''|*[!0-9]*) echo 0 ;; *) echo "$1" ;; esac; }

# ssh to n4 flaps under load, so three attempts before calling it unreadable -- otherwise a single
# reset would burn one of the three escalation strikes.
OUT=""
RC=1
for _try in 1 2 3; do
  OUT=$(ssh -o ConnectTimeout=20 -o BatchMode=yes "$NODE" \
          "docker exec $CTR bash $PROBE" 2>&1)
  RC=$?
  printf '%s' "$OUT" | grep -q '^probe ' && break
  sleep 10
done

if ! printf '%s' "$OUT" | grep -q '^probe '; then
  N=$(( $(num "$(cat "$UNREAD_FILE" 2>/dev/null)") + 1 ))
  echo "$N" > "$UNREAD_FILE"
  SHORT=$(printf '%s' "$OUT" | tr '\n' ' ' | cut -c1-160)
  if [ "$N" -ge 3 ]; then
    echo "$(date) E146REP state=UNREADABLE alive=? traindone=? cells=?/2 done=? fail=- epoch=? note=probe_unreadable_${N}x_FIRE rc=$RC out=$SHORT"
    exit 0
  fi
  echo "$(date) E146REP state=UNREADABLE alive=? traindone=? cells=?/2 done=? fail=- epoch=? note=probe_unreadable_${N}x_quiet rc=$RC out=$SHORT"
  exit 1
fi
echo 0 > "$UNREAD_FILE"

ALIVE=$(num "$(printf '%s' "$OUT" | sed -n 's/.*alive=\([0-9]*\).*/\1/p')")
TD=$(num "$(printf    '%s' "$OUT" | sed -n 's/.*traindone=\([0-9]*\).*/\1/p')")
CELLS=$(num "$(printf '%s' "$OUT" | sed -n 's/.*cells=\([0-9]*\).*/\1/p')")
DONE=$(num "$(printf  '%s' "$OUT" | sed -n 's/.*done=\([0-9]*\).*/\1/p')")
EP=$(printf '%s' "$OUT" | sed -n 's/.*epoch=\(-*[0-9]*\).*/\1/p')
[ -z "$EP" ] && EP="?"
FAIL=$(printf '%s' "$OUT" | sed -n 's/.*fail=\([^ ]*\).*/\1/p')
[ -z "$FAIL" ] && FAIL="-"

if [ "$DONE" -eq 1 ] && [ "$CELLS" -ge 2 ]; then
  echo 0 > "$STALL_FILE"; echo 0 > "$INCON_FILE"
  echo "$(date) E146REP state=DONE alive=$ALIVE traindone=$TD cells=$CELLS/2 done=$DONE fail=$FAIL epoch=$EP note=scorer_wrote_DONE"
  exit 0
fi

# 2026-09-28: THE recurrence killer. The scorer only touches DONE_<run> after both evals exited 0,
# so done=1 with cells<2 is not a state the pipeline can legitimately be in -- it means the probe is
# counting a path the scorer never writes. That exact mismatch (collect dir named e146rep_code vs
# e146_code) held this watcher at state=RUNNING for 5 h with every artifact already on disk, and the
# E144 watcher died the same way. An impossible state must FIRE, never stay quiet: whatever the next
# path bug is, it now costs one poll interval instead of a morning.
if [ "$DONE" -eq 1 ] && [ "$CELLS" -lt 2 ]; then
  I=$(( $(num "$(cat "$INCON_FILE" 2>/dev/null)") + 1 ))
  echo "$I" > "$INCON_FILE"
  if [ "$I" -ge 2 ]; then
    echo "$(date) E146REP state=INCONSISTENT alive=$ALIVE traindone=$TD cells=$CELLS/2 done=$DONE fail=$FAIL epoch=$EP note=DONE_marker_but_cells_short_${I}x_FIRE_check_probe_glob_vs_scorer_write_path"
    exit 0
  fi
  echo "$(date) E146REP state=RUNNING alive=$ALIVE traindone=$TD cells=$CELLS/2 done=$DONE fail=$FAIL epoch=$EP note=possible_cell_glob_mismatch_${I}/2"
  exit 1
fi
echo 0 > "$INCON_FILE"

if [ "$FAIL" != "-" ]; then
  echo 0 > "$STALL_FILE"
  echo "$(date) E146REP state=FAILED alive=$ALIVE traindone=$TD cells=$CELLS/2 done=$DONE fail=$FAIL epoch=$EP note=scorer_wrote_FAIL"
  exit 0
fi

# alive=0 before train_done means the trainer died; two consecutive polls to ride out the gap
# between the trainer exiting and the scorer finishing.
if [ "$ALIVE" -eq 0 ] && [ "$TD" -lt 1 ]; then
  S=$(( $(num "$(cat "$STALL_FILE" 2>/dev/null)") + 1 ))
  echo "$S" > "$STALL_FILE"
  if [ "$S" -ge 2 ]; then
    echo "$(date) E146REP state=STALLED alive=0 traindone=$TD cells=$CELLS/2 done=$DONE fail=$FAIL epoch=$EP note=no_trainer_alive_${S}x_FIRE"
    exit 0
  fi
  echo "$(date) E146REP state=RUNNING alive=0 traindone=$TD cells=$CELLS/2 done=$DONE fail=$FAIL epoch=$EP note=possible_stall_${S}/2"
  exit 1
fi

echo 0 > "$STALL_FILE"
echo "$(date) E146REP state=RUNNING alive=$ALIVE traindone=$TD cells=$CELLS/2 done=$DONE fail=$FAIL epoch=$EP note=ok"
exit 1
