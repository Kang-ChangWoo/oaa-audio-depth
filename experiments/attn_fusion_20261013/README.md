# experiments/attn_fusion_20261013

Six default-off variants of the OAA ray-cross-attention / observation-fusion / loss / masking,
plus the tests that have to pass before any of them is trained.

## What was added to the release code

| file | change |
|---|---|
| `model/oaa.py` | `RayMicAttn`: `qk_norm`, `ray_gate`, `view_fusion ∈ {global,hier,hier-geo}`; optional `obs_valid` masking in `RayMicAttn`, `InterMicAttn` and `_fine_lift`; `collect_stats` diagnostics |
| `core/metrics.py` | `depth_loss(D, gt, mask, kind, wlat)` with `kind ∈ {plain, spherical}` |
| `train_oaa.py` | `--qk-norm --ray-gate --view-fusion --depth-loss --explicit-obs-mask --dump-attn-stats`, the validity mask under `--subset-aug`, `manifest.json`, `attn_stats.jsonl` |
| `core/ckpt.py` | rebuilds the variants from the checkpoint's own args; absent keys fall back to the reference |
| `tests/test_oaa_attention_variants.py` | CPU/float64 contract + equivalence suite (15 tests) |
| `tests/gpu_smoke_attn_fusion.py` | bf16 autocast, peak VRAM, step/inference timing |

**Every new parameter is created only when its flag is on.** A default-built model therefore has
the reference state-dict key set, and an original checkpoint loads with `strict=True`.

## Two pre-existing bugs fixed along the way

Both were in the released `model/oaa.py` and both are no-ops at float32 and under bf16 autocast:

1. `_pose_tensors` and the ear-axis vector in the geometry bias built `torch.tensor(...)` without
   `dtype=`, so the module could not run in float64 at all — which is exactly the precision an
   equivalence proof needs. Now they inherit the surrounding dtype.
2. Internal `.float()` calls would have *downcast* a float64 tensor, capping the hierarchical-fusion
   equivalence check at 1e-7. Replaced with a promote-only helper (`_hiprec`).

## Running it

```bash
source env.local.sh
bash experiments/attn_fusion_20261013/commands.sh tests
bash experiments/attn_fusion_20261013/commands.sh screen 0 E1
```

Results go **outside** the repository, to `/root/local1/changwoo/attn_fusion_20261013/`, with
`PREREG.md`, `REPORT_attn_fusion.md` and `results/*.json`. The locked hypotheses, the decision
gates, and the reason the screening cell is Replica **r6** rather than r8 are in that `PREREG.md`.

## Two things that will bite you

- **The README's headline recipe is not the champion recipe.** `configs/replica/oaa_r8_fin.json`
  trained with `warmup_ep: 8.0`, not the `--warmup-ep 4` in the README and in `CLAUDE.md` §2. Take
  recipes from `configs/<dataset>/<run>.json` key by key.
- **`oaa_r8_fin` is the only Replica champion with `subset_aug: true`**, and that cell's measured
  run-to-run spread on best val MAE is ~4 %. Screening a <1 % effect there cannot work; see
  `PREREG.md` §1.

## Equivalence results that gate the training (float64, CPU, 15/15 passing)

| claim | measured |
|---|---|
| default path vs pristine pre-change module, N=2 and N=8 | max\|diff\| **0.0** (fp32, exact) |
| `RayMicAttn` default path, forward + Q/K/V/`bias_mlp` grads, N=2/4/6/8 | max\|diff\| **0.0** |
| `hier` vs `global` output, 16 configs | **1.78e-15** (double eps 2.2e-16) |
| `hier` vs `global` gradients, relative to the largest gradient | **3.13e-16** |
| `hier` vs a plain mean over observations (the forbidden implementation) | **4.0e-2** — not a mean |
| `ray_gate` at init vs ungated | **0.0** |
| `hier-geo` at init vs `hier` | **0.0** |
| `obs_valid` all-True vs `None`, all three fusion modes | **0.0** |
| perturbing a masked-out observation's spectrogram | **0.0** (unmasked control: 5.4e-1) |
| dropped observation's share of the outer attention mass | **exactly 0**, kept views sum to 1.0 |
| checkpoint round-trip through `core.ckpt.build`, 6 configs | **0.0** |

Parameter cost at the production width (C=256, 8 heads, 15.354 M total): `--qk-norm` **+16**,
`--ray-gate` **+2056**, `--view-fusion hier` **+0**, `--view-fusion hier-geo` **+904**.
