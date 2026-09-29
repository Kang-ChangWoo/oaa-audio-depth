#!/usr/bin/env python3
"""E142-fix FRAC sensitivity read-out + proof that adding --frac left the evaluator inert."""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path

B = Path("/root/local1/changwoo/e142fix")
G = B / "eval_results"
BINS = ["overall", "<0.5", "0.5-1.5", "1.5-4", ">4"]
SEEDS = ("0", "1", "2")


def ld(d, c):
    p = d / f"{c}.json"
    return json.loads(p.read_text()) if p.exists() else None


def mean(d, c, b, stat="f1"):
    v = []
    for k in SEEDS:
        j = ld(d, f"{c}_s{k}")
        if j and j["agg"].get(b):
            v.append(j["agg"][b][stat])
    v = [x for x in v if not math.isnan(x)]
    return st.mean(v) if v else float("nan")


def paired_t(d, b):
    """Per-sequence (r8 N=2) - (r2 N=8) pooled over selection seeds."""
    diffs = []
    for k in SEEDS:
        a, c = ld(d, f"r8_N2_s{k}"), ld(d, f"r2_N8_s{k}")
        if not a or not c:
            continue
        for s in sorted(set(a["per_seq"]) & set(c["per_seq"])):
            x, y = a["per_seq"][s].get(b), c["per_seq"][s].get(b)
            if x and y and not (math.isnan(x["f1"]) or math.isnan(y["f1"])):
                diffs.append(x["f1"] - y["f1"])
    if len(diffs) < 2:
        return None
    m = st.mean(diffs); sd = st.stdev(diffs)
    return {"n": len(diffs), "mean": m, "t": m / (sd / math.sqrt(len(diffs))) if sd else float("nan"),
            "n_pos": sum(1 for x in diffs if x > 0)}


print("=" * 100)
print("INERTNESS PROOF -- the grid (FRAC hard-coded 0.25) vs a re-run with --frac 0.25")
print("=" * 100)
worst = 0.0
for c in ("r8_N2", "r2_N8"):
    for b in BINS:
        a, z = mean(G, c, b), mean(B / "frac/f0.25", c, b)
        if not (math.isnan(a) or math.isnan(z)):
            worst = max(worst, abs(a - z))
        print(f"  {c:<8}{b:<10}{a:.6f}  {z:.6f}   diff {a-z:+.2e}")
print(f"  worst |diff| = {worst:.2e}  ->  {'PASS' if worst < 1e-12 else 'FAIL'}")

print()
print("=" * 100)
print("FRAC SENSITIVITY, FIXED REFERENCE, SLICE A (budget 8).  FRAC 0.25 is the registered primary;")
print("0.50 and 1.00 are a labelled robustness check of the top-FRAC rule, not a re-judgement.")
print("=" * 100)
print(f"{'FRAC':<7}{'bin':<10}{'R=r8N2':>9}{'T=r2N8':>9}{'delta':>9}"
      f"{'R_P':>8}{'R_R':>8}{'T_P':>8}{'T_R':>8}{'pair n':>8}{'p.mean':>9}{'t':>8}{'Rwin':>9}")
for fr in ("0.25", "0.50", "1.00"):
    d = B / f"frac/f{fr}"
    for b in ("overall", ">4"):
        R, T = mean(d, "r8_N2", b), mean(d, "r2_N8", b)
        pt = paired_t(d, b)
        row = (f"{fr:<7}{b:<10}{R:>9.4f}{T:>9.4f}{R-T:>+9.4f}"
               f"{mean(d,'r8_N2',b,'precision'):>8.4f}{mean(d,'r8_N2',b,'recall'):>8.4f}"
               f"{mean(d,'r2_N8',b,'precision'):>8.4f}{mean(d,'r2_N8',b,'recall'):>8.4f}")
        if pt:
            row += f"{pt['n']:>8}{pt['mean']:>+9.4f}{pt['t']:>8.2f}{pt['n_pos']:>5}/{pt['n']:<3}"
        print(row)
    print()

print("=" * 100)
print("READ-OUT: does the sign of the slice-A verdict depend on FRAC?")
print("=" * 100)
for b in ("overall", ">4"):
    signs = []
    for fr in ("0.25", "0.50", "1.00"):
        d = B / f"frac/f{fr}"
        signs.append(mean(d, "r8_N2", b) - mean(d, "r2_N8", b))
    same = all(x < 0 for x in signs) or all(x > 0 for x in signs)
    print(f"  {b:<10} deltas " + "  ".join(f"{x:+.4f}" for x in signs)
          + f"   -> sign {'STABLE' if same else 'FLIPS'} across FRAC")
