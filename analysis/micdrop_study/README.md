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
