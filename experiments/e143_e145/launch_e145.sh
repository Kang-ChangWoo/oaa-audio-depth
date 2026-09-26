#!/bin/bash
# E145 launcher: one tmux window per training run + one scorer window per run, inside the
# container's `claude` session so Changwoo can attach to any of them. No nohup, no bare background.
# usage: launch_e145.sh "<gpu>:<run>:<cue>:<lossweight>:<seed>" ...
set -u
export E145_PY=${E145_PY:-/root/storage/envs/shared_audio/bin/python}
export FORCE_BOTH=${FORCE_BOTH:-ctrl_s0}
CODE=/root/storage/e145_code
OUT=/root/local1/changwoo/e145
mkdir -p $OUT/out $OUT/logs $OUT/results $OUT/eval_results $OUT/pred
for spec in "$@"; do
  IFS=":" read -r GPU RUN CUE LW SEED <<< "$spec"
  W=e145_g${GPU}
  SW=e145_sc_${RUN}
  tmux kill-window -t claude:$W 2>/dev/null
  tmux kill-window -t claude:$SW 2>/dev/null
  tmux new-window -t claude -n $W -d
  tmux send-keys -t claude:$W "export E145_PY=$E145_PY; bash $CODE/run_worker_e145.sh $GPU '$RUN:$CUE:$LW:$SEED'" C-m
  tmux new-window -t claude -n $SW -d
  tmux send-keys -t claude:$SW "export E145_PY=$E145_PY; export FORCE_BOTH=$FORCE_BOTH; bash $CODE/score_e145.sh $RUN $GPU" C-m
  echo "launched $RUN (cue=$CUE lw=$LW seed=$SEED) on gpu$GPU  train=$W scorer=$SW"
done
