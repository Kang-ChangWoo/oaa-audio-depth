#!/usr/bin/env python3
"""The decisive framing: spend 4x the observation budget on HEADINGS vs on POSITIONS.

Both comparisons below are "4x the observations", differing only in which axis the extra
observations are spent on, and both are scored on the fixed reference:

  headings : r8 N=8 (32 obs, 8 positions x 4 headings)  vs  r2 N=8 (8 obs, 8 positions x 1 heading)
  positions: r2 N=8 (8 obs, 8 positions x 1 heading)    vs  r2 N=2 (2 obs, 2 positions x 1 heading)

Same multiplier, same reference, same selection seeds.
"""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path

FIX = Path("/root/local1/changwoo/e142fix/eval_results")
SEEDS = ("0", "1", "2")
BINS = ("overall", "<0.5", "0.5-1.5", "1.5-4", ">4")


def ld(c):
    p = FIX / f"{c}.json"
    return json.loads(p.read_text()) if p.exists() else None


def mean(c, b, stat="f1"):
    v = []
    for k in SEEDS:
        j = ld(f"{c}_s{k}")
        if j and j["agg"].get(b):
            v.append(j["agg"][b][stat])
    v = [x for x in v if not math.isnan(x)]
    return st.mean(v) if v else float("nan")


def paired(A, T, b):
    diffs = []
    for k in SEEDS:
        a, c = ld(f"{A}_s{k}"), ld(f"{T}_s{k}")
        if not a or not c:
            continue
        for s in sorted(set(a["per_seq"]) & set(c["per_seq"])):
            x, y = a["per_seq"][s].get(b), c["per_seq"][s].get(b)
            if x and y and not (math.isnan(x["f1"]) or math.isnan(y["f1"])):
                diffs.append(x["f1"] - y["f1"])
    if len(diffs) < 2:
        return None
    m = st.mean(diffs); sd = st.stdev(diffs)
    return {"n": len(diffs), "mean": m,
            "t": m / (sd / math.sqrt(len(diffs))) if sd else float("nan"),
            "n_pos": sum(1 for x in diffs if x > 0)}


CASES = [("headings 4x", "r8_N8", "r2_N8"), ("positions 4x", "r2_N8", "r2_N2")]

print("=" * 100)
print("WHERE TO SPEND 4x THE OBSERVATION BUDGET  (fixed reference, FRAC 0.25, seeds 0/1/2)")
print("=" * 100)
print(f"{'axis':<14}{'bin':<10}{'before':>9}{'after':>9}{'gain':>9}"
      f"{'dP':>9}{'dR':>9}{'pair n':>8}{'t':>8}{'win':>9}")
for name, A, T in CASES:
    for b in BINS:
        aft, bef = mean(A, b), mean(T, b)
        if math.isnan(aft) or math.isnan(bef):
            continue
        dP = mean(A, b, "precision") - mean(T, b, "precision")
        dR = mean(A, b, "recall") - mean(T, b, "recall")
        s = paired(A, T, b)
        row = (f"{name:<14}{b:<10}{bef:>9.4f}{aft:>9.4f}{aft-bef:>+9.4f}{dP:>+9.4f}{dR:>+9.4f}")
        if s:
            row += f"{s['n']:>8}{s['t']:>8.2f}{s['n_pos']:>4}/{s['n']:<3}"
        print(row)
    print()

print("=" * 100)
print("RATIO OF THE TWO SPENDS (positions gain / headings gain), same 4x multiplier")
print("=" * 100)
for b in BINS:
    h = mean("r8_N8", b) - mean("r2_N8", b)
    p = mean("r2_N8", b) - mean("r2_N2", b)
    if math.isnan(h) or math.isnan(p) or abs(h) < 1e-9:
        continue
    print(f"  {b:<10} headings {h:+.4f}   positions {p:+.4f}   positions/headings = {p/h:6.2f}x")
