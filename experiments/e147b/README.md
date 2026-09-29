# E147b — `FRAC` sensitivity of the E147 near/far trade

Landed 2026-09-27. Results stay off the repo, under `/root/local1/changwoo/e147b/`.
**Zero GPU** (GPU 0–7 were all held by E146 and the ZInD control; nothing was touched.)
63 new cells in **3 min 24 s**.

## What it measured

E147 buys far-field F1 by spending the budget on position, but pays for it near the camera. That
drop could be an artifact of the `FRAC` keep-fraction in the scorer rather than a real trade. This
re-scores the frozen E147 ladders at `frac 0.50` (control) and `frac 1.00` (trend arm) against the
registered `frac 0.25` baseline. Pre-registered rule: **artifact if the near-field drop shrinks to
≤ half**.

## Verdict

**The trade is REAL, and the artifact hypothesis is refuted with the wrong sign** — the drop does
not shrink, it **grows** about 1.5× with a bigger budget:

| ladder | `D(0.25)` | `D(0.50)` | `R = D(0.50)/D(0.25)` |
|---|---|---|---|
| **L32** (primary, 8 seqs) | +.0633 | **+.0992** | **1.568** |
| L24 (confirmatory, 15 seqs) | +.0543 | **+.0836** | **1.539** |

`D(f) = F1_{<0.5}(N=8) − F1_{<0.5}(N=top)`, M1 scoring, `r2`, seeds 0/1/2, cohorts frozen to E147.
`frac 1.00` continues the same direction (`D` = +.1261 L32, +.1042 L24). M1 and M2 agree on
**every** cell, and paired support strengthens with budget
(L32: t=−2.83 p=1.2e−2 at 0.25 → t=−6.77 p=3.3e−6 at 0.50 → t=−7.51 p=8.6e−7 at 1.00).

**Mechanism: the near-field collapse is 100 % precision — recall is pinned at 1.0000.** More
positions flood the near field with false positives; nothing is being lost.

## Caveat that must travel with E147

E147's headline ("`>4 m` does not saturate by N=32") **survives at `frac 0.50`** but **dies at
`frac 1.00`**. Quote the no-saturation result with its `frac` attached.

## Layout

- `e147b_run_frac.sh` — the 63-cell `frac` sweep over the frozen E147 cohorts.
- `e147b_agg.py` — aggregation, M1/M2 agreement, the `R` ratio against the pre-registered rule,
  and the precision/recall decomposition.
- `plot_e147b.py` — the `frac` sensitivity figure.
