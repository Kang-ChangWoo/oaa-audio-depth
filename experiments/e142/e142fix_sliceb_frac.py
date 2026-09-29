#!/usr/bin/env python3
"""Slice B' (matched view count = 8, identical positions) under the FRAC sweep, fixed reference.

This is the comparison where the two arms occupy the SAME eight positions, so coverage is identical
and the only difference is the number of headings per position.  If the rotation advantage is real it
has to survive here and it has to survive the top-FRAC rule.
"""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path

B = Path("/root/local1/changwoo/e142fix")
SEEDS = ("0", "1", "2")
FRACS = ("0.25", "0.50", "1.00")


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


def paired(d, A, T, b):
    diffs = []
    for k in SEEDS:
        a, c = ld(d, f"{A}_s{k}"), ld(d, f"{T}_s{k}")
        if not a or not c:
            continue
        for s in sorted(set(a["per_seq"]) & set(c["per_seq"])):
            x, y = a["per_seq"][s].get(b), c["per_seq"][s].get(b)
            if x and y and not (math.isnan(x["f1"]) or math.isnan(y["f1"])):
                diffs.append(x["f1"] - y["f1"])
    if len(diffs) < 2:
        return None
    m = st.mean(diffs)
    sd = st.stdev(diffs)
    return {"n": len(diffs), "mean": m,
            "t": m / (sd / math.sqrt(len(diffs))) if sd else float("nan"),
            "n_pos": sum(1 for x in diffs if x > 0)}


HDR = ("FRAC", "bin", "R", "T", "delta", "R_P", "R_R", "T_P", "T_R", "pair n", "t", "Rwin")
print("=" * 104)
print("SLICE B' (matched view count = 8, IDENTICAL positions) under the FRAC sweep, fixed reference")
print("   R = r8 N=8 : 4 headings at each of 8 positions      T = r2 N=8 : 1 heading at the same 8")
print("   coverage identical by construction -> the only variable is headings per position")
print("=" * 104)
print(f"{HDR[0]:<7}{HDR[1]:<10}{HDR[2]:>9}{HDR[3]:>9}{HDR[4]:>9}"
      f"{HDR[5]:>8}{HDR[6]:>8}{HDR[7]:>8}{HDR[8]:>8}{HDR[9]:>8}{HDR[10]:>8}{HDR[11]:>9}")
for fr in FRACS:
    d = B / f"frac/f{fr}"
    for b in ("overall", "<0.5", "0.5-1.5", "1.5-4", ">4"):
        R, T = mean(d, "r8_N8", b), mean(d, "r2_N8", b)
        if math.isnan(R) or math.isnan(T):
            continue
        s = paired(d, "r8_N8", "r2_N8", b)
        row = (f"{fr:<7}{b:<10}{R:>9.4f}{T:>9.4f}{R-T:>+9.4f}"
               f"{mean(d,'r8_N8',b,'precision'):>8.4f}{mean(d,'r8_N8',b,'recall'):>8.4f}"
               f"{mean(d,'r2_N8',b,'precision'):>8.4f}{mean(d,'r2_N8',b,'recall'):>8.4f}")
        if s:
            row += f"{s['n']:>8}{s['t']:>8.2f}{s['n_pos']:>5}/{s['n']:<3}"
        print(row)
    print()

print("SIGN STABILITY ACROSS FRAC")
for b in ("overall", "<0.5", "0.5-1.5", "1.5-4", ">4"):
    ds = [mean(B / f"frac/f{fr}", "r8_N8", b) - mean(B / f"frac/f{fr}", "r2_N8", b) for fr in FRACS]
    if any(math.isnan(x) for x in ds):
        continue
    ok = all(x > 0 for x in ds) or all(x < 0 for x in ds)
    print(f"  {b:<10} " + "  ".join(f"{x:+.4f}" for x in ds)
          + f"   -> {'STABLE' if ok else 'FLIPS'}")
