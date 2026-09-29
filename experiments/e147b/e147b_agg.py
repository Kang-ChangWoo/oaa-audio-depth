#!/usr/bin/env python3
"""E147b aggregation: FRAC sensitivity of the E147 position ladders (PREREG E147b sec 3-4).

NEW FILE.  Reads only per-sequence JSONs written by the unmodified `e142_eval_fixedref.py`;
never re-scores, never touches E142/E147 code or the registered FRAC=0.25 grid.

Cohorts are frozen to E147's, read from the existing `r2_N0_s0` record (geometry, frac-independent).
Scoring rules M1 (primary) / M2 (secondary) are copied verbatim from `e147_agg.py`.
"""
from __future__ import annotations
import json, statistics
from pathlib import Path
import numpy as np
from scipy import stats

GRID = Path("/root/local1/changwoo/e142fix/eval_results")          # registered FRAC=0.25
FRACD = Path("/root/local1/changwoo/e142fix/e147b_frac")           # f0.25 / f0.50 / f1.00
OUT = Path("/root/local1/changwoo/e147b")
BINS = ("overall", "<0.5", "0.5-1.5", "1.5-4", ">4")
SEEDS = ("0", "1", "2")
LADDERS = {"L24": (24, [1, 2, 4, 8, 16, 24]), "L32": (32, [1, 2, 4, 8, 16, 32])}
FRACS = ("0.25", "0.50", "1.00")

# prereg sec 3, fixed before the numbers
D_BASE = {"L32": 0.5882 - 0.5249, "L24": 0.6287 - 0.5744}
R_ARTIFACT, R_REAL = 0.50, 0.80
# prereg sec 4: a step is saturated only if BOTH absolute and relative gates fire
SAT_ABS, SAT_REL = 0.005, 0.10
E142_PUBLISHED = [0.0041, 0.0095, 0.0252, 0.0524, 0.0931]   # N=1,2,4,8,16 full-39 M2
E142_PUBLISHED_ALL = 0.1113


def load(src: Path, cell: str):
    f = src / f"{cell}.json"
    return json.loads(f.read_text()) if f.exists() else None


def seq_value(m, bin_name, stat, rule):
    b = m[bin_name]
    v = b[stat]
    if v is None:
        return None
    if not np.isnan(v):
        return float(v)
    if rule == "M2":
        return None
    if b["n_ref"] > 0 and b["n_pred"] == 0:
        return 0.0 if stat in ("f1", "recall") else None
    return None


def cell_mean(cd, cohort, bin_name, stat, rule):
    vals, missing = [], 0
    for s in cohort:
        m = cd["per_seq"].get(s)
        if m is None:
            missing += 1
            continue
        v = seq_value(m, bin_name, stat, rule)
        if v is not None:
            vals.append(v)
    return (float(np.mean(vals)) if vals else None), len(vals), missing


def per_seq_vector(cd, cohort, bin_name, stat, rule):
    out = {}
    for s in cohort:
        m = cd["per_seq"].get(s)
        if m is None:
            continue
        v = seq_value(m, bin_name, stat, rule)
        if v is not None:
            out[s] = v
    return out


def seed_mean(src, mode, N, cohort, bin_name, stat, rule):
    """Mean over seeds of the cohort mean; also returns (pstdev, any_missing)."""
    per_seed, miss_any = [], False
    for sd in SEEDS:
        cd = load(src, f"{mode}_N{N}_s{sd}")
        if cd is None:
            miss_any = True
            continue
        v, _, miss = cell_mean(cd, cohort, bin_name, stat, rule)
        if miss:
            miss_any = True
        if v is not None:
            per_seed.append(v)
    if not per_seed:
        return None, 0.0, True
    sd_ = statistics.pstdev(per_seed) if len(per_seed) > 1 else 0.0
    return float(np.mean(per_seed)), float(sd_), miss_any


def paired(src, mode, a, b, cohort, bin_name, rule):
    pa, pb = [], []
    for sd in SEEDS:
        ca, cb = load(src, f"{mode}_N{a}_s{sd}"), load(src, f"{mode}_N{b}_s{sd}")
        if not ca or not cb:
            continue
        va = per_seq_vector(ca, cohort, bin_name, "f1", rule)
        vb = per_seq_vector(cb, cohort, bin_name, "f1", rule)
        for s in sorted(set(va) & set(vb)):
            pa.append(va[s]); pb.append(vb[s])
    if len(pa) > 1 and np.std(np.array(pb) - np.array(pa)) > 0:
        t, p = stats.ttest_rel(pb, pa)
        return float(t), float(p), len(pa)
    return float("nan"), float("nan"), len(pa)


def src_for(frac):
    return GRID if frac == "0.25" else FRACD / f"f{frac}"


def main():
    L, P = [], None
    L2 = L.append
    base = load(GRID, "r2_N0_s0")
    T = {s: m["_ref"]["n_cam_all"] for s, m in base["per_seq"].items()}
    all39 = sorted(T)
    cohorts = {"L24": sorted(s for s in all39 if T[s] >= 24),
               "L32": sorted(s for s in all39 if T[s] >= 32)}

    L2("=" * 118)
    L2("E147b — FRAC SENSITIVITY OF THE E147 POSITION LADDERS   (fixed reference, CPU only, no GPU)")
    L2("=" * 118)
    L2(f"cohorts frozen to E147: L24 n={len(cohorts['L24'])} (T>=24) | L32 n={len(cohorts['L32'])} (T>=32)")
    L2(f"prereg: R = D(0.50)/D(0.25);  R<=({R_ARTIFACT}) or D<=0 -> artifact ; R>={R_REAL} -> real ; between -> partial")
    L2(f"prereg baselines  D(0.25): L32 = {D_BASE['L32']:+.4f}   L24 = {D_BASE['L24']:+.4f}")
    L2(f"saturation gate (repaired): abs gain <= {SAT_ABS:+.3f} AND relative gain <= {SAT_REL:.0%}  (both)")
    L2("")

    gates = {}

    # ---------------- G1: aggregator reproduces the published E142 full-39 M2 ladder -------------
    L2("#" * 118)
    L2("# G1  integrity: does this aggregator reproduce the PUBLISHED E142 full-39 M2 ladder?")
    L2("#" * 118)
    got = []
    for N in [1, 2, 4, 8, 16]:
        v, _, _ = seed_mean(GRID, "r2", N, all39, ">4", "f1", "M2")
        got.append(v)
    cdall = load(GRID, "r2_N0_s0")
    vals = [seq_value(m, ">4", "f1", "M2") for m in cdall["per_seq"].values()]
    got_all = float(np.mean([v for v in vals if v is not None]))
    worst = max(abs(g - e) for g, e in zip(got, E142_PUBLISHED))
    worst = max(worst, abs(got_all - E142_PUBLISHED_ALL))
    for N, g, e in zip([1, 2, 4, 8, 16], got, E142_PUBLISHED):
        L2(f"   N={N:<3} got {g:.4f}  published {e:.4f}   diff {g-e:+.1e}")
    L2(f"   all   got {got_all:.4f}  published {E142_PUBLISHED_ALL:.4f}   diff {got_all-E142_PUBLISHED_ALL:+.1e}")
    gates["G1"] = worst < 5e-5
    L2(f"   worst |diff| = {worst:.2e}  ->  {'PASS' if gates['G1'] else 'FAIL'}")
    L2("")

    # ---------------- G2: --frac 0.25 re-run is inert vs the hard-coded grid ---------------------
    L2("#" * 118)
    L2("# G2  inertness: `--frac 0.25` re-scored into a fresh dir vs the hard-coded FRAC=0.25 grid")
    L2("#" * 118)
    worst2, n_cmp = 0.0, 0
    for N in [1, 2, 4, 8, 16, 24, 32]:
        for sd in SEEDS:
            a = load(GRID, f"r2_N{N}_s{sd}")
            b = load(FRACD / "f0.25", f"r2_N{N}_s{sd}")
            if not a or not b:
                continue
            for s in sorted(set(a["per_seq"]) & set(b["per_seq"])):
                for bn in BINS:
                    va, vb = a["per_seq"][s][bn]["f1"], b["per_seq"][s][bn]["f1"]
                    if va is None or vb is None:
                        continue
                    if np.isnan(va) and np.isnan(vb):
                        continue
                    n_cmp += 1
                    worst2 = max(worst2, abs(float(va) - float(vb)))
    gates["G2"] = worst2 == 0.0
    L2(f"   compared {n_cmp} per-sequence per-bin F1 values across 21 cells")
    L2(f"   worst |diff| = {worst2:.2e}  ->  {'PASS' if gates['G2'] else 'FAIL'}")
    L2("")

    # ---------------- G3: completeness -----------------------------------------------------------
    L2("#" * 118)
    L2("# G3  completeness of every ladder cell at every frac")
    L2("#" * 118)
    ok3 = True
    for frac in FRACS:
        src = src_for(frac)
        for lname, (_, ns) in LADDERS.items():
            for N in ns:
                for sd in SEEDS:
                    cd = load(src, f"r2_N{N}_s{sd}")
                    if cd is None:
                        L2(f"   MISSING frac={frac} {lname} N={N} s{sd}")
                        ok3 = False
                        continue
                    miss = [s for s in cohorts[lname] if s not in cd["per_seq"]]
                    if miss:
                        L2(f"   frac={frac} {lname} N={N} s{sd} missing {len(miss)} cohort seqs")
                        ok3 = False
    gates["G3"] = ok3
    L2(f"   -> {'PASS' if ok3 else 'FAIL'}   (cohort sizes {len(cohorts['L24'])} / {len(cohorts['L32'])})")
    L2("")

    # ---------------- main tables ----------------------------------------------------------------
    curves = {}
    for lname, (nmax, ns) in LADDERS.items():
        cohort = cohorts[lname]
        for rule in ("M1", "M2"):
            L2("#" * 118)
            L2(f"# ladder={lname} (T>={nmax}, n={len(cohort)} fixed)   scoring {rule}"
               + ("  (primary: n_pred==0 -> F1 0)" if rule == "M1" else "  (secondary: nan-exclusion)"))
            L2("#" * 118)
            for frac in FRACS:
                src = src_for(frac)
                L2(f"--- frac = {frac}" + ("   [registered primary]" if frac == "0.25"
                                           else "   [comparison arm]" if frac == "0.50"
                                           else "   [trend arm]"))
                L2(f"{'N':>4} " + " ".join(f"{b:>16}" for b in BINS)
                   + f" {'<.5 P':>8} {'<.5 R':>8} {'>4 P':>8} {'>4 R':>8}")
                for N in ns:
                    row = []
                    for b in BINS:
                        mu, sd_, _ = seed_mean(src, "r2", N, cohort, b, "f1", rule)
                        row.append("nan" if mu is None else f"{mu:.4f}+-{sd_:.4f}")
                        curves[(lname, rule, frac, b, N)] = mu
                    pr = []
                    for b in ("<0.5", ">4"):
                        for stat in ("precision", "recall"):
                            mu, _, _ = seed_mean(src, "r2", N, cohort, b, stat, rule)
                            pr.append("nan" if mu is None else f"{mu:.4f}")
                    L2(f"{N:>4} " + " ".join(f"{r:>16}" for r in row)
                       + " " + " ".join(f"{x:>8}" for x in pr))
                L2("")

    # ---------------- Q1: near-field trade verdict (prereg sec 3) --------------------------------
    L2("#" * 118)
    L2("# Q1  NEAR/FAR TRADE — is the `<0.5 m` peak-to-top drop a FRAC budget artifact?")
    L2("#" * 118)
    verdicts = {}
    for rule in ("M1", "M2"):
        L2(f"--- scoring {rule}")
        for lname, (nmax, ns) in LADDERS.items():
            cohort, top = cohorts[lname], ns[-1]
            L2(f"  {lname}:  D(f) = F1_<0.5(N=8) - F1_<0.5(N={top})")
            Ds = {}
            for frac in FRACS:
                src = src_for(frac)
                a = curves[(lname, rule, frac, "<0.5", 8)]
                b = curves[(lname, rule, frac, "<0.5", top)]
                D = a - b
                Ds[frac] = D
                t, p, n = paired(src, "r2", 8, top, cohort, "<0.5", rule)
                L2(f"    frac={frac}   F1(8)={a:.4f}  F1({top})={b:.4f}   D={D:+.4f}"
                   f"   paired t={t:+.2f} p={p:.1e} n={n}")
            base_D = D_BASE[lname] if rule == "M1" else Ds["0.25"]
            R = Ds["0.50"] / base_D
            if Ds["0.50"] <= 0 or R <= R_ARTIFACT:
                v = "metric artifact"
            elif R >= R_REAL:
                v = "real trade"
            else:
                v = "partial"
            verdicts[(rule, lname)] = (v, R, Ds)
            L2(f"    prereg baseline D(0.25)={base_D:+.4f}   R = D(0.50)/D(0.25) = {R:+.3f}"
               f"   -> {v.upper()}")
            L2(f"    trend arm frac=1.00: D={Ds['1.00']:+.4f}  "
               f"(monotone shrink with budget: {'YES' if Ds['1.00'] < Ds['0.50'] < Ds['0.25'] else 'NO'})")
            L2("")
    same = len({verdicts[("M1", l)][0] for l in LADDERS})
    q1 = verdicts[("M1", "L32")][0] if same == 1 else "partial"
    m1m2 = all(verdicts[("M1", l)][0] == verdicts[("M2", l)][0] for l in LADDERS)
    L2(f"  L32 (primary) -> {verdicts[('M1','L32')][0]} ; L24 (confirmatory) -> {verdicts[('M1','L24')][0]}")
    L2(f"  prereg: ladders disagree -> 'partial'.  M1 vs M2 agree on every cell: {'YES' if m1m2 else 'NO'}")
    L2(f"  => Q1 VERDICT: {q1.upper()}" + ("" if m1m2 else "   [M1/M2 DISAGREE -> UNRESOLVED per prereg sec 2]"))
    L2("")

    # ---------------- Q2: robustness of E147's no-saturation conclusion -------------------------
    L2("#" * 118)
    L2("# Q2  does E147's `>4 m` no-saturation conclusion survive frac 0.50?  (prereg sec 4)")
    L2("#" * 118)
    q2 = {}
    for rule in ("M1", "M2"):
        L2(f"--- scoring {rule}")
        for lname, (nmax, ns) in LADDERS.items():
            cohort, top = cohorts[lname], ns[-1]
            for frac in FRACS:
                src = src_for(frac)
                c = {N: curves[(lname, rule, frac, ">4", N)] for N in ns}
                steps = []
                sat_at = None
                for a, b in zip(ns, ns[1:]):
                    d = c[b] - c[a]
                    rel = d / c[a] if c[a] and c[a] > 0 else float("inf")
                    t, p, n = paired(src, "r2", a, b, cohort, ">4", rule)
                    fired = (d <= SAT_ABS) and (rel <= SAT_REL)
                    if fired and sat_at is None:
                        sat_at = a
                    steps.append((a, b, d, rel, t, p, n, fired))
                mono = all(s[2] >= 0 for s in steps)
                fin_d, fin_rel = steps[-1][2], steps[-1][3]
                surv = (fin_d > 0) and (fin_rel >= SAT_REL) and mono
                q2[(rule, lname, frac)] = (surv, fin_d, fin_rel, sat_at, mono)
                L2(f"  {lname} frac={frac}: " + "  ".join(f"{a}->{b} {d:+.4f}({rel*100:+.0f}%)"
                                                          for a, b, d, rel, *_ in steps))
                L2(f"     monotone={mono}  final {ns[-2]}->{top}: d={fin_d:+.4f} rel={fin_rel*100:+.1f}%"
                   f" t={steps[-1][4]:+.2f} p={steps[-1][5]:.1e}"
                   f"  first saturated step at N={sat_at}"
                   f"  -> criterion {'SURVIVES' if surv else 'FAILS'}")
            L2("")
    q2v = all(q2[(r, l, "0.50")][0] for r in ("M1", "M2") for l in LADDERS)
    L2(f"  => Q2 VERDICT at frac 0.50: E147's no-saturation conclusion "
       f"{'SURVIVES' if q2v else 'FAILS'} (all ladders x both rules)")
    L2("")

    L2("#" * 118)
    L2("# GATES")
    L2("#" * 118)
    for k in ("G1", "G2", "G3"):
        L2(f"  {k}: {'PASS' if gates[k] else 'FAIL'}")
    L2(f"  G4 (md5) checked in logs/run_frac.log, not here")
    L2("")
    L2("#" * 118)
    L2(f"# ONE LINE: the near/far trade is {q1.upper()}; E147's no-saturation-by-32 "
       f"{'SURVIVES' if q2v else 'FAILS'} at frac 0.50.")
    L2("#" * 118)

    txt = "\n".join(L)
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "results" / "AGG_E147b.txt").write_text(txt)
    (OUT / "results" / "verdict_e147b.json").write_text(json.dumps({
        "cohorts": {k: v for k, v in cohorts.items()},
        "gates": gates,
        "prereg_D_base": D_BASE,
        "R_bounds": {"artifact_max": R_ARTIFACT, "real_min": R_REAL},
        "Q1": {f"{r}|{l}": {"verdict": verdicts[(r, l)][0], "R": verdicts[(r, l)][1],
                            "D": verdicts[(r, l)][2]}
               for r in ("M1", "M2") for l in LADDERS},
        "Q1_verdict": q1, "M1_M2_agree": m1m2,
        "Q2": {f"{r}|{l}|{f}": {"survives": q2[(r, l, f)][0], "final_delta": q2[(r, l, f)][1],
                                "final_rel": q2[(r, l, f)][2], "first_sat_N": q2[(r, l, f)][3],
                                "monotone": q2[(r, l, f)][4]}
               for r in ("M1", "M2") for l in LADDERS for f in FRACS},
        "Q2_verdict_survives_at_050": q2v,
    }, indent=1))
    (OUT / "results" / "plotdata_e147b.json").write_text(json.dumps({
        "ladders": {k: v[1] for k, v in LADDERS.items()},
        "curves": {f"{l}|{r}|{f}|{b}|{N}": curves[(l, r, f, b, N)]
                   for (l, r, f, b, N) in curves},
    }, indent=1))
    print(txt)


if __name__ == "__main__":
    main()
