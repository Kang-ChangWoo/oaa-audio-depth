#!/usr/bin/env python3
"""E142 aggregation: slice A (matched observation budget) and slice B (matched view count).

Verdicts follow PREREG_E142.md sec 4 / 4-1 verbatim. Nothing here chooses a threshold.
"""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path

OUT = Path("/root/local1/changwoo/e142")
ER = OUT / "eval_results"
DIAG = OUT / "diag"
BINS = ["overall", "<0.5", "0.5-1.5", "1.5-4", ">4"]
HEADINGS = {"r2": 1, "r8": 4}
SEEDS = ["0", "1", "2"]


def load(mode, N, seed):
    p = ER / f"{mode}_N{N}_s{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cellf1(d, b):
    a = d["agg"].get(b)
    return (a["f1"], a["n_seq"]) if a else (float("nan"), 0)


def mean_sd(v):
    v = [x for x in v if not math.isnan(x)]
    if not v:
        return float("nan"), float("nan"), 0
    return st.mean(v), st.pstdev(v), len(v)


def paired_seq(dR, dT, b):
    """Per-sequence R-T over the sequences both arms scored for bin b."""
    common = set(dR["per_seq"]) & set(dT["per_seq"])
    diffs = []
    for k in sorted(common):
        a, c = dR["per_seq"][k].get(b), dT["per_seq"][k].get(b)
        if a and c and not (math.isnan(a["f1"]) or math.isnan(c["f1"])):
            diffs.append(a["f1"] - c["f1"])
    if not diffs:
        return None
    m = st.mean(diffs)
    sd = st.stdev(diffs) if len(diffs) > 1 else 0.0
    t = m / (sd / math.sqrt(len(diffs))) if sd > 0 else float("nan")
    return {"n": len(diffs), "mean": m, "sd": sd, "t": t,
            "n_pos": sum(1 for x in diffs if x > 0)}


def grid_table():
    print("=" * 108)
    print("GRID  (f1, mean +- sd over selection seeds 0/1/2 ; 'sp' = linspace max-spread, single run)")
    print("=" * 108)
    hdr = f"{'cell':<12}{'budget':>7}" + "".join(f"{b:>16}" for b in BINS) + f"{'n_seq(ov/>4)':>14}"
    print(hdr)
    for N in [1, 2, 4, 8, 16]:
        for mode in ["r2", "r8"]:
            B = N * HEADINGS[mode]
            ds = [load(mode, N, s) for s in SEEDS]
            ds = [d for d in ds if d]
            if not ds:
                continue
            line = f"{mode+' N='+str(N):<12}{B:>7}"
            nseqs = ""
            for b in BINS:
                vals = [cellf1(d, b)[0] for d in ds]
                m, sd, n = mean_sd(vals)
                line += f"  {m:.4f}+-{sd:.4f}" if n else f"{'--':>16}"
                if b == "overall":
                    nseqs += str(cellf1(ds[0], b)[1])
                if b == ">4":
                    nseqs += "/" + str(cellf1(ds[0], b)[1])
            print(line + f"{nseqs:>14}")
        print("-" * 108)
    print()
    print(f"{'cell (sp)':<12}{'budget':>7}" + "".join(f"{b:>10}" for b in BINS))
    for N in [1, 2, 4, 8, 16]:
        for mode in ["r2", "r8"]:
            d = load(mode, N, "sp")
            if not d:
                continue
            print(f"{mode+' N='+str(N):<12}{N*HEADINGS[mode]:>7}"
                  + "".join(f"{cellf1(d,b)[0]:>10.4f}" for b in BINS))


def compare(labelR, R, labelT, T, title):
    """R, T are (mode, N). Returns dict of per-bin seed-mean deltas + paired stats."""
    print()
    print("#" * 100)
    print(f"# {title}")
    print(f"#   rotation arm  R = {labelR}   (budget {R[1]*HEADINGS[R[0]]})")
    print(f"#   translation T = {labelT}   (budget {T[1]*HEADINGS[T[0]]})")
    print("#" * 100)
    out = {}
    print(f"{'bin':<10}{'R':>10}{'T':>10}{'delta':>10}{'seedsign':>10}"
          f"{'pair n':>8}{'pair mean':>11}{'pair t':>9}{'R wins':>9}")
    for b in BINS:
        Rv = [cellf1(d, b)[0] for s in SEEDS if (d := load(*R, s))]
        Tv = [cellf1(d, b)[0] for s in SEEDS if (d := load(*T, s))]
        mR, _, nR = mean_sd(Rv)
        mT, _, nT = mean_sd(Tv)
        if not nR or not nT:
            continue
        per_seed = [r - t for r, t in zip(Rv, Tv) if not (math.isnan(r) or math.isnan(t))]
        sign = f"{sum(1 for x in per_seed if x > 0)}/{len(per_seed)}"
        ps = paired_seq(load(*R, "0"), load(*T, "0"), b)
        out[b] = {"R": mR, "T": mT, "delta": mR - mT, "seed_sign": sign, "paired_seed0": ps}
        if ps:
            print(f"{b:<10}{mR:>10.4f}{mT:>10.4f}{mR-mT:>+10.4f}{sign:>10}"
                  f"{ps['n']:>8}{ps['mean']:>+11.4f}{ps['t']:>9.2f}{ps['n_pos']:>5}/{ps['n']}")
        else:
            print(f"{b:<10}{mR:>10.4f}{mT:>10.4f}{mR-mT:>+10.4f}{sign:>10}{'--':>8}")
    return out


def diagnostics():
    print()
    print("=" * 78)
    print("GEOMETRY / REFERENCE-SIZE DIAGNOSTICS  (selections are mode-independent)")
    print("=" * 78)
    print(f"{'N':>4}{'seed':>6}{'n_seq':>7}{'mean n_used':>12}{'extent(m)':>11}"
          f"{'nn spacing':>12}{'mean ref vox':>14}")
    for N in [1, 2, 4, 8, 16]:
        for s in ["0", "1", "2", "sp"]:
            p = DIAG / f"N{N}_s{s}.json"
            if not p.exists():
                continue
            a = json.loads(p.read_text())["agg"]
            rv = a.get("mean_n_ref_vox")
            print(f"{N:>4}{s:>6}{a['n_sequences']:>7}{a['mean_n_used']:>12.2f}"
                  f"{a['mean_extent_m']:>11.3f}{a['mean_nn_spacing_m']:>12.3f}"
                  + (f"{rv:>14.0f}" if rv else f"{'--':>14}"))


def verdict(sliceA, sliceB):
    print()
    print("=" * 78)
    print("VERDICT (PREREG sec 4 / 4-1 applied mechanically)")
    print("=" * 78)
    o = sliceA["overall"]; f = sliceA[">4"]
    ps = o["paired_seed0"]
    cond_mag_A = o["delta"] >= 0.010
    cond_mag_B = o["delta"] <= -0.010
    seed33 = o["seed_sign"] in ("3/3", "0/3")
    pair_ok = ps and ps["n_pos"] >= math.ceil(2 * ps["n"] / 3)
    print(f"slice A  ROT_O = {o['delta']:+.4f}   seed sign {o['seed_sign']}   "
          f"paired {ps['n_pos']}/{ps['n']} (need >= {math.ceil(2*ps['n']/3)})")
    print(f"slice A  ROT_F = {f['delta']:+.4f}   seed sign {f['seed_sign']}")
    if cond_mag_A and o["seed_sign"] == "3/3" and pair_ok:
        v = "A = 회전 우위"
    elif cond_mag_B and o["seed_sign"] == "0/3" and ps and ps["n_pos"] <= ps["n"] - math.ceil(2*ps["n"]/3):
        v = "B = 이동 우위"
    else:
        if abs(f["delta"]) >= 0.020 and f["seed_sign"] in ("3/3", "0/3"):
            v = "C1 = 전체 F1 동등, 원거리에서만 갈림 (" + ("회전" if f["delta"] > 0 else "이동") + " 우위)"
        else:
            v = "C2 = 미결 -> slice B 대조군으로 착지"
    print(f"  ==> slice A verdict: {v}")

    ob = sliceB["overall"]
    psb = ob["paired_seed0"]
    print(f"slice B  ROT_O = {ob['delta']:+.4f}   seed sign {ob['seed_sign']}   "
          f"paired {psb['n_pos']}/{psb['n']}")
    print(f"slice B  ROT_F = {sliceB['>4']['delta']:+.4f}   seed sign {sliceB['>4']['seed_sign']}")
    return v


if __name__ == "__main__":
    grid_table()
    A = compare("r8 N=2", ("r8", 2), "r2 N=8", ("r2", 8),
                "SLICE A (matched observation budget = 8)  -- the practical question")
    B = compare("r8 N=2", ("r8", 2), "r2 N=2", ("r2", 2),
                "SLICE B (matched view count = 2, identical reference cloud) -- the mechanism question")
    A16 = compare("r8 N=4", ("r8", 4), "r2 N=16", ("r2", 16),
                  "SLICE A' (matched observation budget = 16)")
    B8 = compare("r8 N=8", ("r8", 8), "r2 N=8", ("r2", 8),
                 "SLICE B' (matched view count = 8, identical reference cloud)")
    B1 = compare("r8 N=1", ("r8", 1), "r2 N=1", ("r2", 1),
                 "SLICE B'' (matched view count = 1, single view)")
    B4 = compare("r8 N=4", ("r8", 4), "r2 N=4", ("r2", 4),
                 "SLICE B''' (matched view count = 4)")
    diagnostics()
    verdict(A, B)
