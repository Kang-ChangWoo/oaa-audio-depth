#!/bin/bash
# attn_fusion_20261013 — OAA attention/fusion variants.
#
# INDEXING (Changwoo 2026-10-13 23:59, item 3): every run name, output path and result row from
# this attempt carries the prefix  attnfus_  . Nothing else in this repo uses it: the existing
# families are `0820_*` (campaign, via 0820_train_oaa_afm.py) and `*_fin` (release, via
# train_oaa.py). Do not mix the prefixes, and do not drop it from a run name.
#
#   run name   attnfus_<EXP>_r2_<rep|mp3d>_s<seed>
#   out dir    $OUT/<rep|mp3d>/<run name>/
#   results    $RES/compare_attnfus.json        <- NEVER comparison/compare.json
#   index      $RES/../INDEX_attnfus.tsv
#
# WHY A SEPARATE compare.json: eval.py:43 does `merged.update(res)` into
# <compare-dir>/compare.json, so re-using a run name against the default --compare-dir would
# silently overwrite a cited row. The prefix plus an explicit --compare-dir are both required.
#
# BASELINE SERIES: `_fin` (STFT hop 160, train_oaa.py). See BASELINE_SERIES_AUDIT.md for why, and
# for why the `0820_h44_*` series is a different model rather than a different number.
#
#   source env.local.sh
#   bash experiments/attn_fusion_20261013/commands.sh tests
#   bash experiments/attn_fusion_20261013/commands.sh screen <gpu> <EXP> <rep|mp3d>
set -euo pipefail

E="${E:-$HOME/attn_fusion_20261013}"      # results live OUTSIDE the repository
OUT="${OUT:-$E/runs}"
RES="${RES:-$E/results}"
MODE="${1:-help}"
mkdir -p "$OUT/rep" "$OUT/mp3d" "$RES"

# Interpreter, explicit on purpose. Bare `python` in this container resolves to
# /opt/conda/bin/python (the base env), which HAS torch but NOT soundfile, so every run dies in the
# first dataloader batch with ModuleNotFoundError. The existing launchers (run_point.sh and the
# 0820_* queues) also hard-code this path; do not replace it with `python`.
PY="${PY:-/opt/conda/envs/shared_audio/bin/python}"
[ -x "$PY" ] || { echo "interpreter not found: $PY"; exit 1; }
"$PY" -c "import soundfile, torch" 2>/dev/null || { echo "$PY lacks soundfile/torch"; exit 1; }

# ---- 2-observation champion recipes, copied key-by-key from the runs' own train_done.json args.
# configs/replica/oaa_r2_fin.json : nviews 2, r2, 40 ep, bs 24, accum(absent)=1, lr 5e-4, warmup 4,
#                                  seed 0, dim 256, no subset_aug, no stem_stride1.  best val 0.3330
# configs/mp3d/oaa_r2_fin.json    : identical except the data module.                 best val 1.06396
# Identical across both datasets, which is why "per dataset" costs no extra recipe work.
# --seed is appended per run. seed 0 is the ORIGINAL's seed, so attnfus_E0_*_s0 is a direct
# reproduction of oaa_r2_fin; seeds 1 and 2 are new and exist to measure the replication floor.
R2_ARGS="--nviews 2 --data-mode r2 --epochs 40 --batch-size 24 --accum 1 --lr 5e-4 --warmup-ep 4 \
         --dim 256"

variant_flags() {
  case "$1" in
    E0)     echo "" ;;
    E1)     echo "--qk-norm" ;;
    E2)     echo "--ray-gate" ;;
    E3)     echo "--qk-norm --ray-gate" ;;
    E4hier) echo "--view-fusion hier" ;;      # bug detector: provably identical to E0
    E4geo)  echo "--view-fusion hier-geo" ;;
    E5)     echo "--depth-loss spherical" ;;
    # ---- round 1 (2026-10-14): structural handles only, L1 loss kept everywhere.
    R1rounds3) echo "--rounds 3" ;;                       # existing flag, no new code
    R1rounds1) echo "--rounds 1" ;;                       # other direction on the same axis
    R1e3r3)    echo "--qk-norm --ray-gate --rounds 3" ;;  # round-0 winner + the far-field arm
    R1nogeo)   echo "--no-geo-bias" ;;                    # diagnostic: is the geometry term load-bearing at 2 obs?
    R1rounds4) echo "--rounds 4" ;;                       # extend the depth axis
    R1e3r1)    echo "--qk-norm --ray-gate --rounds 1" ;;  # winner at the shallow end of the axis
    R1nogeoqk) echo "--no-geo-bias --qk-norm" ;;          # does QK-norm matter with geometry removed?
    R1norayemb) echo "--no-ray-emb" ;;                    # reviewer variant-(f) half: query-side ray embedding
    R1notfpe)  echo "--no-tf-pe" ;;                       # coarse-token TF positional encoding off
    # ---- switch COMBINATIONS of the three live improvement switches (qk-norm, ray-gate, hier-geo).
    # 2^3-1 = 7 subsets; 4 already run (E1 E2 E4geo E3), so exactly 3 are new. No new code.
    C_QH)      echo "--qk-norm --view-fusion hier-geo" ;;
    C_GH)      echo "--ray-gate --view-fusion hier-geo" ;;
    C_QGH)     echo "--qk-norm --ray-gate --view-fusion hier-geo" ;;
    C_QGH_r3)  echo "--qk-norm --ray-gate --view-fusion hier-geo --rounds 3" ;;
    # ---- VALUE changes on pre-existing flags. All verified to build and keep the 256x512 output.
    # lift 32x64 drops one ENCODER downsampling stage (4 -> 3), so the feature maps stay large and
    # micro-batch 24 OOMs at 47.4 GiB. bs6/accum4 is step-matched: same 200 steps, same effective batch.
    V_lift32x64)    echo "--lift-h 32 --lift-w 64 --batch-size 6 --accum 4" ;;   # FAR-FIELD: 4x finer query grid
    V_lift32x64_r3) echo "--lift-h 32 --lift-w 64 --rounds 3 --batch-size 6 --accum 4" ;;
    V_lift8x16)     echo "--lift-h 8 --lift-w 16" ;;       # the other direction on the same axis
    V_stem1)        echo "--stem-stride1" ;;               # FAR-FIELD: no 2x upsample, no interpolation
    V_stem1_r3)     echo "--stem-stride1 --rounds 3" ;;
    V_stem1_r1)     echo "--stem-stride1 --rounds 1" ;;              # shallow end + far-field
    V_lift32x64_r1) echo "--lift-h 32 --lift-w 64 --rounds 1 --batch-size 6 --accum 4" ;;
    V_rounds5)      echo "--rounds 5" ;;
    V_rounds6)      echo "--rounds 6" ;;
    V_dim320)       echo "--dim 320 --batch-size 12 --accum 2" ;;   # step-matched headroom
    V_dim384)       echo "--dim 384 --batch-size 8 --accum 3" ;;    # step-matched headroom
    V_dim192)       echo "--dim 192" ;;
    V_warm2)        echo "--warmup-ep 2" ;;
    V_warm8)        echo "--warmup-ep 8" ;;                # the r8 champion's value
    V_wd5e4)        echo "--wd 5e-4" ;;
    V_wd1e3)        echo "--wd 1e-3" ;;
    V_lr25)         echo "--lr 2.5e-4" ;;
    V_lr1e3)        echo "--lr 1e-3" ;;
    V_bs12a2)       echo "--batch-size 12 --accum 2" ;;     # step-matched: 2nd numerical-noise sample
    # ---- existing ablations, as diagnostics for the reviewer's variant-(f) attack on this cell
    A_noposeemb)    echo "--no-pose-emb" ;;
    # ---- PARAMETER-LEVEL tuning INSIDE the three methods already introduced (Changwoo 16:15).
    # Every one of these is motivated by a number measured in a trained checkpoint, not by a guess:
    #   logit_scale moved only 3-11 % in 40 epochs  -> the init IS the knob
    #   gate mean settled at 1.28-1.39 (>1)         -> the model wants amplification, start it there
    #   geo bias RMS 0.22-1.56 vs content 2.28-5.03 -> the geometry term is too small to route
    #   fine_to_dec RMS 0.0057 vs decoder 0.0173    -> the only high-res path is nearly dead
    # P_lr* tests all four at once: are they LR-starved rather than useless?
    P_lr10)       echo "--qk-norm --ray-gate --param-lr-mult 10" ;;
    P_lr30)       echo "--qk-norm --ray-gate --param-lr-mult 30" ;;
    P_lr10_fine)  echo "--param-lr-mult 10" ;;                      # only fine_to_dec exists here
    P_ls173)      echo "--qk-norm --logit-scale-init 1.733" ;;       # log(sqrt(dk)) = the L2 variant
    P_ls069)      echo "--qk-norm --logit-scale-init 0.693" ;;       # multiplier 2
    P_lsneg)      echo "--qk-norm --logit-scale-init -0.693" ;;      # multiplier 0.5
    P_geo11)      echo "--qk-norm --geo-scale-init 1.0986" ;;        # geometry multiplier 3
    P_geo22)      echo "--qk-norm --geo-scale-init 2.1972" ;;        # geometry multiplier 9
    P_gate_i)     echo "--ray-gate --ray-gate-init 0.6" ;;           # gate starts at 1.29 = measured mean
    P_gatech)     echo "--ray-gate --ray-gate-mode channel" ;;       # per-(ray,channel) gate
    P_gatech_i)   echo "--ray-gate --ray-gate-mode channel --ray-gate-init 0.6" ;;
    P_fine02)     echo "--fine-skip-init 0.02" ;;                    # ~3.5x the trained 0.0057
    P_fine01)     echo "--fine-skip-init 0.01" ;;
    P_all)        echo "--qk-norm --ray-gate --param-lr-mult 10 --fine-skip-init 0.02 --geo-scale-init 1.0986" ;;
    *) echo "UNKNOWN" ;;
  esac
}

case "$MODE" in

tests)
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 "$PY" tests/test_oaa_attention_variants.py
  "$PY" tests/gpu_smoke_attn_fusion.py --nviews 8 --batch 3 --json "$RES/gpu_smoke_r8_b3.json"
  ;;

screen)   # screen <gpu> <EXP> <rep|mp3d> [seed]
  GPU="$2"; EXP="$3"; DS="$4"; SEED="${5:-0}"
  [ "$GPU" = "4" ] && { echo "refusing GPU 4: confirmed faulty 2026-10-13"; exit 1; }
  V="$(variant_flags "$EXP")"; [ "$V" = "UNKNOWN" ] && { echo "unknown EXP $EXP"; exit 1; }
  case "$DS" in
    rep)  DM=data_0422 ;;
    mp3d) DM=data_mp3d ;;
    *) echo "dataset must be rep or mp3d"; exit 1 ;;
  esac
  RUN="attnfus_${EXP}_r2_${DS}_s${SEED}"
  # refuse to clobber our own earlier attempt at the same cell
  [ -e "$OUT/$DS/$RUN/train_done.json" ] && { echo "$RUN already finished; refusing to overwrite"; exit 1; }
  SN="seed$SEED"; [ "$SEED" = "0" ] && SN="seed0(=original)"
  printf '%s\t%s\t%s\t%s\tgpu%s\t%s\tSTARTED\t\t%s\n' "$RUN" "$EXP" "$DS" "r2/$DM" "$GPU" \
      "$(date -u '+%F %T')Z" "$SN" >> "$E/INDEX_attnfus.tsv"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" DATA_MODULE=$DM \
  "$PY" train_oaa.py --run-name "$RUN" $R2_ARGS --seed "$SEED" $V --dump-attn-stats --out-dir "$OUT/$DS"
  printf '%s\t%s\t%s\t%s\tgpu%s\t%s\tFINISHED\t%s\t%s\n' "$RUN" "$EXP" "$DS" "r2/$DM" "$GPU" \
      "$(date -u '+%F %T')Z" "$("$PY" -c "import json;print(json.load(open('$OUT/$DS/$RUN/train_done.json'))['best_val_mae_m'])" 2>/dev/null || echo '?')" \
      "$SN" >> "$E/INDEX_attnfus.tsv"
  ;;

queue)    # queue <gpu> <rep|mp3d> <spec...>   spec = EXP or EXP:seed, run back to back on one card
  GPU="$2"; DS="$3"; shift 3
  for SPEC in "$@"; do
    EXP="${SPEC%%:*}"; SEED="${SPEC#*:}"; [ "$SEED" = "$EXP" ] && SEED=0
    echo "[queue] gpu$GPU $DS $EXP seed$SEED  $(date -u '+%F %T')Z"
    bash "$0" screen "$GPU" "$EXP" "$DS" "$SEED" || echo "[queue] $EXP:$SEED FAILED, continuing"
  done
  ;;

evaltest) # evaltest <rep|mp3d> <run...>   test set, ONCE, at the end, into our own compare file
  DS="$2"; shift 2
  # train_done.json guard: eval.py scores ANY best.pth it finds, including a mid-training one.
  # That is how attnfus_E4hier_r2_mp3d_s0 got a bogus 0.9790 into a results file on 2026-10-14.
  for _r in "$@"; do
    [ -f "$OUT/$DS/$_r/train_done.json" ] || { echo "REFUSING $_r: no train_done.json, run is unfinished"; exit 1; }
  done
  case "$DS" in rep) DM=data_0422 ;; mp3d) DM=data_mp3d ;; *) exit 1 ;; esac
  DATA_MODULE=$DM "$PY" eval.py --run-name "$@" --out-dir "$OUT/$DS" --compare-dir "$RES"
  mv "$RES/compare.json" "$RES/compare_attnfus_${DS}.json" 2>/dev/null || true
  ;;

refscore) # re-score the _fin baseline checkpoints into OUR file, never the cited one
  DS="${2:-rep}"
  case "$DS" in rep) DM=data_0422; CD=comparison ;; mp3d) DM=data_mp3d; CD=comparison_mp3d ;; esac
  DATA_MODULE=$DM "$PY" eval.py --run-name oaa_r2_fin --compare-dir "$RES" --out-dir "$CD"
  mv "$RES/compare.json" "$RES/refscore_fin_${DS}.json" 2>/dev/null || true
  ;;

*) sed -n '1,24p' "$0" ;;
esac
