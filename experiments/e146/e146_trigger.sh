#!/bin/bash
# E146 watcher trigger. Runs ON THE MACMINI, reads n4 over ssh, prints ONE line and sets an exit code.
#
#   exit 0 = FIRE (wake the agent)      exit 1 = QUIET (keep looping)
# The automation schedule is `on-exit until <this script>; do sleep 900; done`, so a 0 stops the
# loop and wakes the session. Every path must set the code deliberately.
#
# Deliberately NOT inline-quoted into an ssh command: the E144 watcher never fired for four hours
# because a nested `sh -c` quoting level broke and `2>/dev/null` hid the error. So:
#   * the remote work lives in a FILE on the node (e146_probe.sh), invoked by name
#   * stderr is KEPT and surfaced in the output line, never discarded
#   * an unreadable probe is an EXPLICIT state, and three in a row FIRES instead of staying silent
#
# Output: one line, 8 whitespace-separated fields.
#   E146 state=<STATE> alive=<distinct runs> traindone=<n> cells=<n>/12 done=<n> fail=<csv|-> note=<text>
# STATE is one of: RUNNING ALL_DONE FAILED STALLED UNREADABLE
set -u
NODE=${E146_NODE:-n4}
CTR=${E146_CTR:-changwoo_audio_n4}
STATEDIR=${E146_STATEDIR:-$HOME/.openclaw/workspace/state}
UNREAD_FILE=$STATEDIR/e146_unreadable_count
STALL_FILE=$STATEDIR/e146_stall_count
mkdir -p "$STATEDIR"

num () { case ${1:-} in ''|*[!0-9]*) echo 0 ;; *) echo "$1" ;; esac; }

# ssh itself can flap (n4 resets connections under load), so give the probe three attempts before
# calling it unreadable -- otherwise a single reset would burn one of the three escalation strikes.
OUT=""
RC=1
for _try in 1 2 3; do
  OUT=$(ssh -o ConnectTimeout=20 -o BatchMode=yes "$NODE" \
          "docker exec $CTR bash /root/storage/e146_code/e146_probe.sh" 2>&1)
  RC=$?
  printf '%s' "$OUT" | grep -q '^probe ' && break
  sleep 10
done

if ! printf '%s' "$OUT" | grep -q '^probe '; then
  N=$(( $(num "$(cat "$UNREAD_FILE" 2>/dev/null)") + 1 ))
  echo "$N" > "$UNREAD_FILE"
  SHORT=$(printf '%s' "$OUT" | tr '\n' ' ' | cut -c1-160)
  if [ "$N" -ge 3 ]; then
    echo "$(date) E146 state=UNREADABLE alive=? traindone=? cells=?/12 done=? fail=- note=probe_unreadable_${N}x_FIRE rc=$RC out=$SHORT"
    exit 0          # "can't read -> don't fire" is the E144 bug; 3 strikes escalates
  fi
  echo "$(date) E146 state=UNREADABLE alive=? traindone=? cells=?/12 done=? fail=- note=probe_unreadable_${N}x_quiet rc=$RC out=$SHORT"
  exit 1
fi
echo 0 > "$UNREAD_FILE"

ALIVE=$(num "$(printf   '%s' "$OUT" | sed -n 's/.*alive=\([0-9]*\).*/\1/p')")
TD=$(num "$(printf      '%s' "$OUT" | sed -n 's/.*traindone=\([0-9]*\).*/\1/p')")
CELLS=$(num "$(printf   '%s' "$OUT" | sed -n 's/.*cells=\([0-9]*\).*/\1/p')")
DONE=$(num "$(printf    '%s' "$OUT" | sed -n 's/.*done=\([0-9]*\).*/\1/p')")
ALLDONE=$(num "$(printf '%s' "$OUT" | sed -n 's/.*alldone=\([01]\).*/\1/p')")
FAIL=$(printf '%s' "$OUT" | sed -n 's/.*fail=\([^ ]*\).*/\1/p')
[ -z "$FAIL" ] && FAIL="-"

if [ "$ALLDONE" -eq 1 ]; then
  echo 0 > "$STALL_FILE"
  echo "$(date) E146 state=ALL_DONE alive=$ALIVE traindone=$TD cells=$CELLS/12 done=$DONE fail=$FAIL note=finisher_touched_E146_ALL_DONE"
  exit 0
fi

if [ "$FAIL" != "-" ]; then
  echo 0 > "$STALL_FILE"
  echo "$(date) E146 state=FAILED alive=$ALIVE traindone=$TD cells=$CELLS/12 done=$DONE fail=$FAIL note=a_scorer_wrote_FAIL"
  exit 0
fi

if [ "$ALIVE" -eq 0 ] && [ "$TD" -lt 6 ]; then
  S=$(( $(num "$(cat "$STALL_FILE" 2>/dev/null)") + 1 ))
  echo "$S" > "$STALL_FILE"
  if [ "$S" -ge 2 ]; then
    echo "$(date) E146 state=STALLED alive=0 traindone=$TD cells=$CELLS/12 done=$DONE fail=$FAIL note=no_trainer_alive_${S}x_FIRE"
    exit 0
  fi
  echo "$(date) E146 state=RUNNING alive=0 traindone=$TD cells=$CELLS/12 done=$DONE fail=$FAIL note=possible_stall_${S}/2"
  exit 1
fi

echo 0 > "$STALL_FILE"
echo "$(date) E146 state=RUNNING alive=$ALIVE traindone=$TD cells=$CELLS/12 done=$DONE fail=$FAIL note=ok"
exit 1
