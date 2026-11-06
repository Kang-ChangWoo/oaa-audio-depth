#!/bin/bash
# Push this node's local experiment artefacts into the NFS experiment dir every 5 minutes, so the
# canonical place to look is the repo path and not my node-local scratch. Shell loop, no agent.
# usage: bash sync_results.sh <n1|n14>
NODE=${1:?usage: sync_results.sh <n1|n14>}
D=/root/storage/implementation/shared_audio/hear360/experiments/20261106_replica_pose_sharing
P=/root/local1/changwoo/poseablate_20261106
E=/root/local1/changwoo/relevel_seedfloor_20261016
mkdir -p "$D/results/$NODE"

while :; do
  cp -f "$P"/results/poseablate_rep_*.json "$D/results/$NODE/" 2>/dev/null
  cp -f "$P"/run_*.log "$D/results/$NODE/" 2>/dev/null
  cp -f "$E"/logs/paperrep_unshared_*.log "$D/results/$NODE/" 2>/dev/null
  cp -f "$E"/PREREG_unsharedenc_20261106.md "$D/" 2>/dev/null

  {
    echo "# STATUS $(date '+%Y-%m-%d %H:%M %Z')  (node $NODE, 5분마다 갱신)"
    echo
    echo "## 학습 (unshared-enc)"
    for f in "$E"/logs/paperrep_unshared_*.log; do
      [ -f "$f" ] || continue
      n=$(grep -c '^\[ep ' "$f")
      last=$(grep '^\[ep ' "$f" | tail -1)
      done_f=$E/runs/rep/$(basename "$f" .log)/train_done.json
      [ -f "$done_f" ] && st=LANDED || st=running
      echo "  $(basename "$f" .log): $n/40  $st  $last"
    done
    echo
    echo "## 교란 채점 패스 (mode-evals)"
    python - <<'PY' 2>/dev/null
import glob, json, os
for f in sorted(glob.glob("/root/local1/changwoo/poseablate_20261106/results/poseablate_rep_*json")):
    d = json.load(open(f))
    n = sum(len([k for k in v if not k.startswith("_")]) for v in d.values())
    print(f"  {os.path.basename(f)}: cells {len(d)}, mode-evals {n}")
PY
    echo
    echo "## 큐 대기 순서 (이 노드)"
    cd "$E" && timeout 150 bash launch_seedfloor.sh --probe 2>/dev/null | grep -E 'READY|SKIP\(running\)' | head -12
  } > "$D/results/STATUS_$NODE.txt" 2>&1

  sleep 300
done
