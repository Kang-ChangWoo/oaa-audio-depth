# E142 — turning vs. walking, and the fixed-reference re-scoring that killed half of it

Landed 2026-09-26. Results stay off the repo, under `/root/local1/changwoo/e142/` and
`/root/local1/changwoo/e142fix/`. Pre-registrations and reports live with the run, not here.

## What it measured

At a fixed observation budget, is it better to spend it on **rotation** (more headings at one
position, `r8`) or on **position** (more positions at one heading, `r2`)? Both axes are separable
in the released eval data: a step is a position, a channel pair is a heading
(`predict.py:37-39`, `predict.py:115-116`). The eval trajectory is a 0.15 m constant-yaw walk, so
`data_0422.py`'s `step//4` "same position" grouping is a **training-side** render convention only
and must not be used to define the position axis.

Grid: mode {`r2`, `r8`} × positions N {1,2,4,8,16} × selection seed {0,1,2,sp}, plus two `N=all`
validity cells. **CPU only, zero training, zero GPU inference** — everything is a re-score of the
released `posterior_{r2,r8}/best.pth` predictions.

## Verdict

**Only walking buys `>4 m`.** At budget 8, fixed reference: rotation `.0045` vs walking `.0524`
(Δ `−.0479`, seed sign 0/3, paired n=30, t = −5.60, rotation wins 0/30). This survives the
byte-identical-reference re-score, so "**position buys the far field**" is a paper claim.

**The "rotation wins overall F1" half is dead.** The original `+.0228` overall-F1 advantage was a
**reference-cloud artifact**: `e114_eval.py` builds the reference as the GT union of the *selected*
steps and bins distance to the *selected* camera positions, so the two arms of the budget-8 slice
were scored against 22,353 vs 43,336 voxels (1.94×). With the reference fixed as a function of
`(scene, seq)` alone, the overall-F1 sign **flips to −.1051**. Never quote the `+.0228`.

## Layout

- `e142_make_subset.py` — budget-controlled prediction subsets (all budget control happens here,
  on the npz side; the evaluator is never edited).
- `e114_eval.py` — the released evaluator, **unmodified** (md5 `5e28675599c7872e34258a1038f503c8`,
  logged before and after every grid). Vendored so the grid pins one exact byte sequence.
- `e142_cell.sh`, `e142_run_all.sh` — one cell, and the 40-cell + 2-gate grid driver (2 min 22 s).
- `e142_agg.py`, `e142_diag.py` — aggregation and the per-slice diagnostics that surfaced the
  reference-cloud confound.

Fixed-reference re-scoring (the correction above):

- `e142_ref_fixed.py` — bakes one reference cloud per `(scene, seq)` from **all** steps at
  `VOXEL/2 = 0.05 m`, so every cell reads the same file regardless of mode and N.
- `e142_eval_fixedref.py` — the evaluator variant that consumes it. Reads `e114_eval.py`, never
  writes it.
- `e142fix_cell.sh`, `e142fix_run_all.sh`, `e142fix_agg.py` — the 42-cell re-score (2 min 21 s).
- `e142fix_frac.sh`, `e142fix_frac_agg.py`, `e142fix_sliceb_frac.py` — `FRAC` sensitivity and the
  view-count-matched slice B.
- `e142fix_pr.py`, `e142fix_spend.py`, `fix_header.py` — precision/recall decomposition, budget
  accounting, report header fixups.
