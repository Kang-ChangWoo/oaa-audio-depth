# E147 — where the position budget saturates

Landed 2026-09-27. Results stay off the repo, under `/root/local1/changwoo/e147/`.
**CPU only — 16 new cells in 45 s, zero GPU** (GPU 0–5 were held by E146 and were not touched.)

## What it measured

E142 established that position, not rotation, buys the far field. E147 asks where that stops:
it extends the position ladder past N=16 to **N=24 and N=32** on cohorts frozen to E142-fix, and
compares the position axis against the rotation axis on the same picture.

The 39 test sequences are 14–60 steps long, so the ladder shortens as it climbs: **N=24 keeps 15
sequences, N=32 keeps 8**. Both ladders are run with the cohort fixed so the comparison stays
paired. The fixed reference (`ref_fixed`) is N-independent and is reused from E142-fix, not
regenerated.

## Verdict

**The position axis does not saturate inside the measurable range.** Pre-registered template **B**:
both cohort-frozen ladders keep climbing to the trajectory ceiling, and the last doubling is the
second-largest gain — L32's 16→32 step is **+0.0461** (paired t = +2.50, p = .020, n = 24).

The rotation axis, on the same picture, saturates at **4–5 headings**. So "spend the observation
budget on position" is now a number, not a sign: the position axis is still alive at 6.4× the
rotation axis's saturation point.

Self-consistency gate passed: replaying the new aggregator under the published M2 (nan-excluded)
rule over all 39 sequences reproduces E142-fix's published ladder **exactly**
(`.0041 / .0095 / .0252 / .0524 / .0931 / .1113`).

## Traps this code encodes

- **`n_pred == 0 -> nan`.** A far bin with no predicted voxels scores as `nan`, not 0. M1 and M2
  differ only in how those cells are folded in; the aggregator computes both and they must agree
  cell-by-cell before a verdict is read.
- The evaluator and subset maker are md5-pinned before and after the grid
  (`e114_eval.py` `5e28675599c7872e34258a1038f503c8`,
  `e142_make_subset.py` `92daefe175a4aaead66e208ec6542b99`,
  `e142_eval_fixedref.py` `13af86bf807a2b8b2be0cfb452fe09bf`).

## Layout

- `e147_run_ladder.sh` — the N=24/32 ladder driver over the frozen cohorts.
- `e147_agg.py` — aggregation, M1/M2 agreement check, paired t on each doubling.
- `plot_e147.py`, `e147_export_plot.py` — the saturation figure and its export.

## Follow-up

The `--frac 0.5` control queued here is E147b, which asks whether the near/far trade is a `FRAC`
artifact. It is not — see `../e147b/`.
