#!/bin/bash
# Fan this node's local artefacts out into the per-experiment results/ dirs in the repo, every 5
# minutes, so each experiment folder is self-contained. Shell loop, no agent. usage: <n1|n14>
NODE=${1:?usage: sync_experiments.sh <n1|n14>}
E=/root/storage/implementation/shared_audio/hear360/experiments
P=/root/local1/changwoo/poseablate_20261106
S=/root/local1/changwoo/relevel_seedfloor_20261016

ALL="20261106_unshared_encoder 20261106_pose_random_heading 20261106_ear_mixing
     20261106_channel_directions 20261106_channel_stacking 20261106_itd_arm"

put() {   # put <src-glob> <exp-dir>
  local d=$E/$2/results/$NODE
  mkdir -p "$d"
  cp -f $1 "$d/" 2>/dev/null
}

while :; do
  put "$P/results/poseablate_rep_*_allmetrics_*.json" 20261106_pose_random_heading
  put "$P/results/poseablate_rep_*_yawrot_*.json"     20261106_pose_random_heading
  put "$P/results/poseablate_rep_*_allmetrics_*.json" 20261106_ear_mixing
  put "$P/results/poseablate_rep_*_earmix_*.json"     20261106_ear_mixing
  put "$P/results/poseablate_rep_*_earleft_*.json"    20261106_ear_mixing
  put "$P/results/poseablate_rep_*_pairbreak_*.json"  20261106_channel_directions
  put "$P/run_*.log"                                  20261106_pose_random_heading
  put "$S/logs/paperrep_unshared_*.log"               20261106_unshared_encoder
  put "$S/logs/paperrep_stack_*.log"                  20261106_channel_stacking
  put "$S/logs/paperrep_roll_*.log"                   20261106_channel_stacking
  put "$S/logs/gibh160_itd_*.log"                     20261106_itd_arm
  cp -f "$S"/PREREG_unsharedenc_20261106.md "$E/20261106_unshared_encoder/" 2>/dev/null
  cp -f "$S"/PREREG_stackall_20261106.md "$E/20261106_channel_stacking/" 2>/dev/null
  cp -f "$S"/PREREG_stackroll_20261106.md "$E/20261106_channel_stacking/" 2>/dev/null

  # exact queue recipe for each experiment's cells, so the folder is reproducible on its own
  grep -n '^paperrep_unshared' "$S/QUEUE.txt" > "$E/20261106_unshared_encoder/results/QUEUE_lines_$NODE.txt" 2>/dev/null
  grep -nE '^paperrep_(stack|roll)' "$S/QUEUE.txt" > "$E/20261106_channel_stacking/results/QUEUE_lines_$NODE.txt" 2>/dev/null
  grep -n '^gibh160_itd' "$S/QUEUE.txt" > "$E/20261106_itd_arm/results/QUEUE_lines_$NODE.txt" 2>/dev/null

  for x in $ALL; do
    f=$E/$x/results/STATUS_$NODE.txt
    {
      echo "# STATUS $(date '+%Y-%m-%d %H:%M %Z')  node=$NODE  (5분마다 갱신)"
      echo
      echo "## 이 노드에서 도는 우리 학습 칸"
      for g in "$S"/logs/paperrep_unshared_*.log "$S"/logs/paperrep_stack_*.log \
               "$S"/logs/paperrep_roll_*.log "$S"/logs/gibh160_itd_*.log; do
        [ -f "$g" ] || continue
        n=$(grep -c '^\[ep ' "$g")
        b=$(basename "$g" .log)
        st=running
        [ -f "$S/runs/rep/$b/train_done.json" ] && st=LANDED
        [ -f "$S/runs/mp3d/$b/train_done.json" ] && st=LANDED
        echo "  $b: $n/40 $st  $(grep '^\[ep ' "$g" | tail -1)"
      done
      echo
      echo "## 교란 채점 패스"
      python - <<'PY' 2>/dev/null
import glob, json, os
for f in sorted(glob.glob("/root/local1/changwoo/poseablate_20261106/results/poseablate_rep_*json")):
    d = json.load(open(f))
    n = sum(len([k for k in v if not k.startswith("_")]) for v in d.values())
    print(f"  {os.path.basename(f)}: cells {len(d)}, mode-evals {n}")
PY
      echo
      echo "## 큐 대기 순서"
      cd "$S" && timeout 150 bash launch_seedfloor.sh --probe 2>/dev/null \
        | grep -E 'READY|SKIP\(running\)' | head -14
    } > "$f" 2>&1
  done
  sleep 300
done
