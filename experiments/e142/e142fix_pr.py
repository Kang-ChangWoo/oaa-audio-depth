#!/usr/bin/env python3
"""E142-fix follow-up, zero extra compute: with the reference held fixed, precision and recall are
directly comparable across cells, so the position effect can be split into

  recall gain    = surfaces the arm did not previously reach at all   (coverage)
  precision gain = surfaces it already reached, now placed correctly  (accuracy / SNR)

Reads only the JSON the fixed-ref grid already wrote.
"""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path

FIX = Path("/root/local1/changwoo/e142fix/eval_results")
ORIG = Path("/root/local1/changwoo/e142/eval_results")
SEEDS = ["0", "1", "2"]
BINS = ["overall", "<0.5", "0.5-1.5", "1.5-4", ">4"]


def load(root, mode, N, seed):
    p = root / f"{mode}_N{N}_s{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cell(root, mode, N, b, stat):
    v = []
    for s in (SEEDS if N else ["0"]):
        d = load(root, mode, N, s)
        if d and d["agg"].get(b):
            v.append(d["agg"][b][stat])
    v = [x for x in v if not math.isnan(x)]
    return st.mean(v) if v else float("nan")


print("=" * 96)
print("P/R LADDER, FIXED REFERENCE -- target identical in every row, so rows are comparable")
print("=" * 96)
for b in (">4", "1.5-4", "overall"):
    print(f"\n  bin {b}")
    print(f"    {'cell':<11}{'F1':>9}{'precision':>11}{'recall':>9}   {'dF1':>8}{'dP':>8}{'dR':>8}  (vs previous row)")
    for mode in ("r2", "r8"):
        prev = None
        for N in (1, 2, 4, 8, 16, 0):
            f = cell(FIX, mode, N, b, "f1")
            p = cell(FIX, mode, N, b, "precision")
            r = cell(FIX, mode, N, b, "recall")
            if math.isnan(f):
                continue
            lab = f"{mode} N=" + ("all" if N == 0 else str(N))
            d = ""
            if prev:
                d = f"   {f-prev[0]:>+8.4f}{p-prev[1]:>+8.4f}{r-prev[2]:>+8.4f}"
            print(f"    {lab:<11}{f:>9.4f}{p:>11.4f}{r:>9.4f}{d}")
            prev = (f, p, r)
        print()

print("=" * 96)
print("SLICE A (budget 8) P/R DECOMPOSITION -- fixed vs original protocol")
print("=" * 96)
print(f"{'bin':<10}{'protocol':<10}{'R(r8 N=2)':>24}{'T(r2 N=8)':>24}")
print(f"{'':<20}{'F1':>8}{'P':>8}{'R':>8}{'F1':>8}{'P':>8}{'R':>8}")
for b in BINS:
    for tag, root in (("fixed", FIX), ("orig", ORIG)):
        row = f"{b if tag=='fixed' else '':<10}{tag:<10}"
        for mode, N in (("r8", 2), ("r2", 8)):
            row += "".join(f"{cell(root, mode, N, b, s):>8.4f}" for s in ("f1", "precision", "recall"))
        print(row)
    print("-" * 68)

print()
print("=" * 96)
print("HOW MUCH OF THE FIXED SCENE EACH ARM EVEN REACHES  (>4 m bin, fixed reference)")
print("=" * 96)
print(f"{'cell':<11}{'>4 recall':>11}{'>4 n_pred (mean/seq)':>24}{'>4 n_ref (mean/seq)':>22}")
for mode in ("r2", "r8"):
    for N in (1, 2, 4, 8, 16, 0):
        d = load(FIX, mode, N, "0")
        if not d:
            continue
        ks = [k for k in d["per_seq"] if d["per_seq"][k].get(">4")]
        npred = st.mean([d["per_seq"][k][">4"]["n_pred"] for k in ks])
        nref = st.mean([d["per_seq"][k][">4"]["n_ref"] for k in ks])
        lab = f"{mode} N=" + ("all" if N == 0 else str(N))
        print(f"{lab:<11}{cell(FIX, mode, N, '>4', 'recall'):>11.4f}{npred:>24.0f}{nref:>22.0f}")
    print()
