# Mic-drop degradation study — trained on 8 mics, tested on 8…1

**Question.** Every model here was trained with all eight mics live. What happens at deployment
when mics die? We zero *k* of the 8 channels at inference (k = 0…7, so 8…1 mics survive), keep
the pose/geometry conditioning truthful, and read the resulting MAE curve. This is the *mic
failure* scenario — a rig that loses microphones — not a smaller rig retrained from scratch.

**Why no-mic-drop models are the fair comparison.** None of the prior-work baselines
(BatVision, EchoScan, EchoDiffusion, Beyond-I2D, ResNet/ViT) uses mic-drop augmentation, so
comparing them against a mic-drop-trained OAA would price the augmentation, not the
architecture. The `novd` group is therefore our side of the like-for-like comparison; the
mic-drop-trained OAA runs are carried separately as a reference for what the augmentation buys.

## Protocol

| | |
|---|---|
| data | Replica, `off3` test split (`DATA_MODULE=data_0422 R0422_SPLIT=off3`) |
| drop | dropped channels zeroed; `view_poses` untouched |
| subsets | 3 fixed-seed random subsets per k (`random.Random(0)`), mean reported; exhaustive C(8,k) is too many |
| metric | latitude-weighted MAE (m), identical accumulator to `eval.py` |
| `ret@1` | MAE(1 mic) / MAE(8 mic) — how much of the 8-mic error survives total rig loss; lower is a flatter curve |

Each model is one data pass evaluating all 22 variants, so the loader dominates runtime.

## Reproduce

```bash
# whole roster, filling free GPUs (<2000MB); resumable, one JSON shard per model
bash analysis/micdrop_study/run.sh
GPUS="3 5" bash analysis/micdrop_study/run.sh      # restrict to specific GPUs
ECHODIFF_PY=/path/to/echodiff/bin/python bash analysis/micdrop_study/run.sh   # include EchoDiffusion

# merge shards -> table.md + curves.json + curve.png  (safe to run mid-sweep)
python3 analysis/micdrop_study/report.py
```

Single model, by hand:

```bash
DATA_MODULE=data_0422 R0422_SPLIT=off3 EVAL_BS=6 CUDA_VISIBLE_DEVICES=3 \
  python3 analysis/micdrop.py --run-name oaa_r8_novdrop --out /tmp/one.json
```

## Files

| file | role |
|---|---|
| `roster.py` | the model list and its three groups — single source of truth for runner and report |
| `run.sh` | dispatches the roster onto empty GPUs, one shard per model, claim-locked so several instances can share the roster |
| `report.py` | merges `shards/*.json` into `table.md`, `curves.json`, `curve.png` |
| `shards/` | per-model raw curves (`curve`, `curve_full` with RMSE/AbsRel/δ, and the exact subsets drawn) |

The curve itself is computed by the repo's existing `analysis/micdrop.py` (extended here to
handle waveform-input models and the `comparison_0820/` run directory); nothing about the
released protocol changed, so previously published mic-drop numbers still reproduce.

## Result (15/15 models, 2026-09-11)

Full numbers in `table.md`, curves in `curves.json`, figure in `curve.png`.

1. **Without mic-drop training nothing degrades gracefully.** Ours and prior work alike lose
   80-95% of their accuracy by 6 live mics; ret@1 lands in 3.1-4.2x for every such model.
2. **With mic-drop training the whole family separates.** All six of our mic-drop-trained runs
   (CNN k<=4 / k<=6 and four AFM variants) sit in a single band well below everything else, and
   the augmentation transfers across backbones: AFM ret@1 goes 3.1-3.3x -> 2.19-2.59x.
3. **Inside that band the ranking splits by where training-time drop reached.** eat+LLRD+cs +vd
   is best from 8 down to 3 mics (0.2291 at 8, the campaign cell record); OAA-CNN +vd(k<=6) wins
   at 2 and 1 mic -- it is the only model that saw two-live-mic states in training (k<=6), every
   other run stopped at k<=4.
4. **Augmentation alone is not enough.** EchoDiffusion's own channel dropout moves it 3.55x ->
   3.24x, about a sixth of the recovery we get; its 6-mic error (0.2915) never returns to its
   own 8-mic level (0.2600). The augmentation pays off on top of position-aware geometric
   attention, not by itself.
5. **ret@1 is not a ranking.** ResNet-18 (3.09x), Beyond-I2D (3.17x) and ViT-B (3.26x) look
   flatter than our un-augmented CNN (4.09x) only because their 8-mic starting point is already
   poor. Judge on absolute MAE at each mic count.
