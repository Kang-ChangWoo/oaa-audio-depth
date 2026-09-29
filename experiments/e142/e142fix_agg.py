#!/usr/bin/env python3
"""E142-fix aggregation: the same slices as e142_agg.py, scored on the FIXED reference,
printed next to the original protocol so the confound's effect is visible per bin.

Verdict wording follows REPORT_E142 sec 4 / PREREG_E142 sec 4 -- no threshold is chosen here.
"""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path

FIX = Path("/root/local1/changwoo/e142fix/eval_results")
ORIG = Path("/root/local1/changwoo/e142/eval_results")
BINS = ["overall", "<0.5", "0.5-1.5", "1.5-4", ">4"]
HEADINGS = {"r2": 1, "r8": 4}
SEEDS = ["0", "1", "2"]


def load(root, mode, N, seed):
    p = root / f"{mode}_N{N}_s{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cellf1(d, b):
    a = d["agg"].get(b)
    return (a["f1"], a["n_seq"]) if a else (float("nan"), 0)


def mean_sd(v):
    v = [x for x in v if not math.isnan(x)]
    if not v:
        return float("nan"), float("nan"), 0
    return st.mean(v), st.pstdev(v), len(v)


def paired(root, R, T, b, seeds=SEEDS):
    """Per-sequence R-T pooled over selection seeds; also the seed-0-only figure that
    REPORT_E142 quoted, so the two protocols can be compared on identical footing."""
    per_seed, all_diffs = {}, []
    for s in seeds:
        dR, dT = load(root, *R, s), load(root, *T, s)
        if not dR or not dT:
            continue
        diffs = []
        for k in sorted(set(dR["per_seq"]) & set(dT["per_seq"])):
            a, c = dR["per_seq"][k].get(b), dT["per_seq"][k].get(b)
            if a and c and not (math.isnan(a["f1"]) or math.isnan(c["f1"])):
                diffs.append(a["f1"] - c["f1"])
        if diffs:
            per_seed[s] = diffs
            all_diffs += diffs

    def stats(d):
        if not d:
            return None
        m = st.mean(d)
        sd = st.stdev(d) if len(d) > 1 else 0.0
        t = m / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
        return {"n": len(d), "mean": m, "sd": sd, "t": t, "n_pos": sum(1 for x in d if x > 0)}

    return {"s0": stats(per_seed.get("0")), "pooled": stats(all_diffs)}


def grid(root, title):
    print("=" * 112)
    print(title)
    print("=" * 112)
    print(f"{'cell':<12}{'budget':>7}" + "".join(f"{b:>16}" for b in BINS) + f"{'n_seq(ov/>4)':>14}")
    for N in [1, 2, 4, 8, 16, 0]:
        for mode in ["r2", "r8"]:
            seeds = SEEDS if N else ["0"]
            ds = [d for s in seeds if (d := load(root, mode, N, s))]
            if not ds:
                continue
            label = f"{mode} N=" + ("all" if N == 0 else str(N))
            bud = "--" if N == 0 else str(N * HEADINGS[mode])
            line = f"{label:<12}{bud:>7}"
            ns = ""
            for b in BINS:
                m, sd, n = mean_sd([cellf1(d, b)[0] for d in ds])
                line += f"  {m:.4f}+-{sd:.4f}" if n else f"{'--':>16}"
                if b == "overall":
                    ns += str(cellf1(ds[0], b)[1])
                if b == ">4":
                    ns += "/" + str(cellf1(ds[0], b)[1])
            print(line + f"{ns:>14}")
        print("-" * 112)


def compare(R, T, title, labelR, labelT):
    print()
    print("#" * 112)
    print(f"# {title}")
    print(f"#   R (rotation)    = {labelR}  budget {R[1]*HEADINGS[R[0]]}")
    print(f"#   T (translation) = {labelT}  budget {T[1]*HEADINGS[T[0]]}")
    print("#" * 112)
    print(f"{'bin':<10} | {'FIXED REF: R':>12}{'T':>9}{'delta':>9}{'seed':>6}"
          f"{'pair n':>7}{'p.mean':>9}{'t':>7}{'Rwin':>8} | "
          f"{'ORIG: delta':>12}{'seed':>6}{'t(s0)':>7}")
    res = {}
    for b in BINS:
        row = {}
        for tag, root in (("fix", FIX), ("orig", ORIG)):
            Rv = [cellf1(d, b)[0] for s in SEEDS if (d := load(root, *R, s))]
            Tv = [cellf1(d, b)[0] for s in SEEDS if (d := load(root, *T, s))]
            mR, _, nR = mean_sd(Rv)
            mT, _, nT = mean_sd(Tv)
            if not nR or not nT:
                row[tag] = None
                continue
            ps = [r - t for r, t in zip(Rv, Tv) if not (math.isnan(r) or math.isnan(t))]
            row[tag] = {"R": mR, "T": mT, "delta": mR - mT,
                        "sign": f"{sum(1 for x in ps if x > 0)}/{len(ps)}",
                        "paired": paired(root, R, T, b)}
        res[b] = row
        f, o = row.get("fix"), row.get("orig")
        if not f:
            continue
        pp = f["paired"]["pooled"] or {"n": 0, "mean": float("nan"), "t": float("nan"), "n_pos": 0}
        os_ = (o["paired"]["s0"] if o and o["paired"]["s0"] else None)
        print(f"{b:<10} | {f['R']:>12.4f}{f['T']:>9.4f}{f['delta']:>+9.4f}{f['sign']:>6}"
              f"{pp['n']:>7}{pp['mean']:>+9.4f}{pp['t']:>7.2f}{pp['n_pos']:>4}/{pp['n']:<3} | "
              + (f"{o['delta']:>+12.4f}{o['sign']:>6}" if o else f"{'--':>12}{'--':>6}")
              + (f"{os_['t']:>7.2f}" if os_ else f"{'--':>7}"))
    return res


def gate():
    print()
    print("=" * 112)
    print("SELF-CONSISTENCY GATE -- at N=all the selected set IS the full set, so fixedref must")
    print("reproduce e114_eval.py exactly.  Any nonzero difference means the new file diverges.")
    print("=" * 112)
    print(f"{'cell':<10}{'bin':<10}{'fixedref':>12}{'e114_eval':>12}{'diff':>14}")
    worst = 0.0
    for mode in ["r2", "r8"]:
        df, do = load(FIX, mode, 0, 0), load(ORIG, mode, 0, 0)
        if not df or not do:
            print(f"{mode:<10}{'--':<10}{'MISSING':>12}")
            continue
        for b in BINS:
            a, c = cellf1(df, b)[0], cellf1(do, b)[0]
            d = a - c
            if not math.isnan(d):
                worst = max(worst, abs(d))
            print(f"{mode+' N=all':<10}{b:<10}{a:>12.6f}{c:>12.6f}{d:>+14.2e}")
    print(f"\n  worst |diff| = {worst:.3e}   -> {'PASS' if worst < 1e-9 else 'FAIL'}")
    return worst


def ref_identity():
    print()
    print("=" * 112)
    print("REFERENCE BYTE-IDENTITY -- md5 of the reference file each cell loaded, per sequence.")
    print("A single md5 per sequence across all 42 cells == the scoring target is byte-identical.")
    print("=" * 112)
    per_seq = {}
    ncells = 0
    for N in [1, 2, 4, 8, 16, 0]:
        for s in (SEEDS + ["sp"] if N else ["0"]):
            for mode in ["r2", "r8"]:
                d = load(FIX, mode, N, s)
                if not d:
                    continue
                ncells += 1
                for k, v in d["ref_md5"].items():
                    per_seq.setdefault(k, set()).add(v)
    bad = {k: v for k, v in per_seq.items() if len(v) != 1}
    print(f"  cells inspected      : {ncells}")
    print(f"  sequences covered    : {len(per_seq)}")
    print(f"  sequences with >1 md5: {len(bad)}  -> {'PASS' if not bad else 'FAIL ' + str(bad)}")
    return len(bad) == 0


def ref_growth():
    print()
    print("=" * 112)
    print("WHAT THE FIX REMOVED -- n_ref on the scoring side, original vs fixed, seed 0")
    print("=" * 112)
    print(f"{'cell':<12}{'orig n_ref (mean/seq)':>24}{'fixed n_ref (mean/seq)':>24}"
          f"{'orig >4 n_ref':>15}{'fixed >4 n_ref':>16}")
    for N in [1, 2, 4, 8, 16, 0]:
        for mode in ["r2", "r8"]:
            df, do = load(FIX, mode, N, "0"), load(ORIG, mode, N, "0")
            if not df or not do:
                continue
            com = sorted(set(df["per_seq"]) & set(do["per_seq"]))
            if not com:
                continue
            mo = st.mean([do["per_seq"][k]["overall"]["n_ref"] for k in com])
            mf = st.mean([df["per_seq"][k]["overall"]["n_ref"] for k in com])
            fo = st.mean([do["per_seq"][k][">4"]["n_ref"] for k in com])
            ff = st.mean([df["per_seq"][k][">4"]["n_ref"] for k in com])
            label = f"{mode} N=" + ("all" if N == 0 else str(N))
            print(f"{label:<12}{mo:>24.0f}{mf:>24.0f}{fo:>15.0f}{ff:>16.0f}")


if __name__ == "__main__":
    grid(FIX, "FIXED-REFERENCE GRID  (f1, mean +- pstdev over selection seeds 0/1/2)")
    A = compare(("r8", 2), ("r2", 8),
                "SLICE A (matched observation budget = 8) -- the paper's claim",
                "r8 N=2", "r2 N=8")
    A16 = compare(("r8", 4), ("r2", 16),
                  "SLICE A' (matched observation budget = 16)", "r8 N=4", "r2 N=16")
    B = compare(("r8", 2), ("r2", 2),
                "SLICE B (matched view count = 2)", "r8 N=2", "r2 N=2")
    B8 = compare(("r8", 8), ("r2", 8),
                 "SLICE B' (matched view count = 8)", "r8 N=8", "r2 N=8")
    w = gate()
    ok = ref_identity()
    ref_growth()
    print()
    print("=" * 112)
    print("HEADLINE")
    print("=" * 112)
    for name, res in (("budget 8 ", A), ("budget 16", A16)):
        f = res[">4"]["fix"]; o = res[">4"]["orig"]
        fo = res["overall"]["fix"]; oo = res["overall"]["orig"]
        pp = f["paired"]["pooled"]
        print(f"  {name}  >4m : fixed R={f['R']:.4f} T={f['T']:.4f} d={f['delta']:+.4f} "
              f"(seed {f['sign']}, pooled t={pp['t']:.2f}, n={pp['n']})"
              f"   | orig d={o['delta']:+.4f} (seed {o['sign']})")
        print(f"  {name}  overall: fixed R={fo['R']:.4f} T={fo['T']:.4f} d={fo['delta']:+.4f} "
              f"(seed {fo['sign']})   | orig d={oo['delta']:+.4f} (seed {oo['sign']})")
    print(f"\n  gate worst |diff| = {w:.3e} ; reference byte-identity = {'PASS' if ok else 'FAIL'}")
