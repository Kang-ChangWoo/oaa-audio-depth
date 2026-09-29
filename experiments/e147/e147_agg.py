#!/usr/bin/env python3
"""E147 aggregation: cohort-fixed position ladders (PREREG E147 sec 2-5).

NEW FILE.  Reads the per-sequence JSONs already written by the unmodified
`e142_eval_fixedref.py`; it never re-scores anything and never touches E142 code.

Two scoring rules, both reported (prereg M1 / M2):
  M1 (primary)   n_pred(bin)==0 with n_ref(bin)>0  ->  F1 = 0, recall = 0, precision = nan
  M2 (secondary) the published nan-exclusion rule

Two cohort-fixed ladders (prereg C1), plus the cohort-confounded full-39 ladder (C3):
  L24 = sequences with T >= 24 (15), N = 1 2 4 8 16 24
  L32 = sequences with T >= 32 (8),  N = 1 2 4 8 16 32
"""
from __future__ import annotations
import json, statistics
from pathlib import Path
import numpy as np
from scipy import stats

EVAL = Path("/root/local1/changwoo/e142fix/eval_results")
OUT = Path("/root/local1/changwoo/e147")
BINS = ("overall", "<0.5", "0.5-1.5", "1.5-4", ">4")
SEEDS = ("0", "1", "2")
LADDERS = {"L24": (24, [1, 2, 4, 8, 16, 24]), "L32": (32, [1, 2, 4, 8, 16, 32])}
SAT_EPS = 0.005  # prereg sec 4, fixed in advance


def load(cell: str):
    f = EVAL / f"{cell}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def seq_value(m, bin_name, stat, rule):
    """One sequence's metric under scoring rule M1 or M2.  None means 'excluded'."""
    b = m[bin_name]
    v = b[stat]
    if v is None:
        return None
    if not np.isnan(v):
        return float(v)
    # nan: why?
    if rule == "M2":
        return None
    # M1: a bin with reference mass but no prediction is a real zero for f1/recall
    if b["n_ref"] > 0 and b["n_pred"] == 0:
        return 0.0 if stat in ("f1", "recall") else None
    return None


def cell_mean(cell_data, cohort, bin_name, stat, rule):
    """Mean over the cohort; returns (value, n_used, n_missing_from_cell)."""
    vals, missing = [], 0
    for s in cohort:
        m = cell_data["per_seq"].get(s)
        if m is None:
            missing += 1
            continue
        v = seq_value(m, bin_name, stat, rule)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, 0, missing
    return float(np.mean(vals)), len(vals), missing


def per_seq_vector(cell_data, cohort, bin_name, stat, rule):
    out = {}
    for s in cohort:
        m = cell_data["per_seq"].get(s)
        if m is None:
            continue
        v = seq_value(m, bin_name, stat, rule)
        if v is not None:
            out[s] = v
    return out


def main():
    base = load("r2_N0_s0")
    T = {s: m["_ref"]["n_cam_all"] for s, m in base["per_seq"].items()}
    all39 = sorted(T)
    cohorts = {"L24": sorted(s for s in all39 if T[s] >= 24),
               "L32": sorted(s for s in all39 if T[s] >= 32),
               "FULL39": all39}

    report, L = {}, []
    P = L.append
    P("=" * 110)
    P("E147 — POSITION-COUNT SATURATION, COHORT-FIXED LADDERS   (fixed reference, CPU only, no GPU)")
    P("=" * 110)
    P(f"cohorts: L24 n={len(cohorts['L24'])} (T>=24) | L32 n={len(cohorts['L32'])} (T>=32) | FULL39 n=39")
    P(f"saturation rule (pre-registered): next ladder step gains <= +{SAT_EPS:.3f} on '>4' F1, M1 scoring")
    P("")

    for mode in ("r2", "r8"):
        tag = "PRIMARY" if mode == "r2" else "SECONDARY"
        for lname, (nmax, ns) in LADDERS.items():
            cohort = cohorts[lname]
            P("#" * 110)
            P(f"# {tag}  mode={mode}  ladder={lname}  (T >= {nmax}, n_seq = {len(cohort)} fixed)")
            P("#" * 110)
            for rule in ("M1", "M2"):
                P(f"--- scoring {rule} " + ("(primary: n_pred==0 -> F1 0)" if rule == "M1"
                                            else "(secondary: published nan-exclusion)"))
                P(f"{'N':>4} " + " ".join(f"{b:>16}" for b in BINS) +
                  f" {'>4 P':>8} {'>4 R':>8} {'n>4':>5} {'void':>5}")
                curve = {}
                for N in ns:
                    row, ok = [], True
                    for b in BINS:
                        per_seed = []
                        for sd in SEEDS:
                            cd = load(f"{mode}_N{N}_s{sd}")
                            if cd is None:
                                ok = False
                                break
                            v, nu, miss = cell_mean(cd, cohort, b, "f1", rule)
                            if miss:
                                ok = False
                            if v is not None:
                                per_seed.append(v)
                        if not per_seed:
                            row.append(("nan", 0))
                            continue
                        mu = float(np.mean(per_seed))
                        sd_ = float(statistics.pstdev(per_seed)) if len(per_seed) > 1 else 0.0
                        row.append((f"{mu:.4f}+-{sd_:.4f}", mu))
                        if b == ">4":
                            curve[N] = mu
                    pr = []
                    for stat in ("precision", "recall"):
                        vv = []
                        for sd in SEEDS:
                            cd = load(f"{mode}_N{N}_s{sd}")
                            if cd is None:
                                continue
                            v, _, _ = cell_mean(cd, cohort, ">4", stat, rule)
                            if v is not None:
                                vv.append(v)
                        pr.append(f"{np.mean(vv):.4f}" if vv else "nan")
                    cd0 = load(f"{mode}_N{N}_s0")
                    n4 = len(per_seq_vector(cd0, cohort, ">4", "f1", rule)) if cd0 else 0
                    P(f"{N:>4} " + " ".join(f"{r[0]:>16}" for r in row) +
                      f" {pr[0]:>8} {pr[1]:>8} {n4:>5} {'VOID' if not ok else '':>5}")
                # linspace arm, reported separately (prereg sec 3)
                spv = []
                for N in ns:
                    cd = load(f"{mode}_N{N}_ssp") or load(f"{mode}_N{N}_s0")
                    v, _, _ = cell_mean(cd, cohort, ">4", "f1", rule) if cd else (None, 0, 0)
                    spv.append("nan" if v is None else f"{v:.4f}")
                P(f"  sp >4 f1 (linspace, not averaged in): " + "  ".join(f"N{n}={x}" for n, x in zip(ns, spv)))

                if rule == "M1":
                    report.setdefault(mode, {})[lname] = dict(curve)
                    P("")
                    P(f"  saturation test on '>4' F1 ({rule}, cohort-fixed):")
                    sat = None
                    for a, b in zip(ns, ns[1:]):
                        if a not in curve or b not in curve:
                            continue
                        d = curve[b] - curve[a]
                        # paired support test over (sequence x seed)
                        pa, pb = [], []
                        for sd in SEEDS:
                            ca, cb = load(f"{mode}_N{a}_s{sd}"), load(f"{mode}_N{b}_s{sd}")
                            if not ca or not cb:
                                continue
                            va = per_seq_vector(ca, cohort, ">4", "f1", rule)
                            vb = per_seq_vector(cb, cohort, ">4", "f1", rule)
                            for s in sorted(set(va) & set(vb)):
                                pa.append(va[s]); pb.append(vb[s])
                        if len(pa) > 1 and np.std(np.array(pb) - np.array(pa)) > 0:
                            t, p = stats.ttest_rel(pb, pa)
                        else:
                            t, p = float("nan"), float("nan")
                        fired = d <= SAT_EPS
                        if fired and sat is None:
                            sat = a
                        mult = f"x{b/a:.2g}"
                        P(f"    {a:>3} -> {b:<3} ({mult:>4})  delta={d:+.4f}  "
                          f"paired t={t:+.2f} p={p:.1e} n={len(pa):>3}  "
                          f"{'SATURATED at N=' + str(a) if fired else 'still paying'}")
                    verdict = (f"saturates at N={sat}" if sat is not None
                               else f"does NOT saturate within measured range (up to N={ns[-1]})")
                    P(f"  => {lname} {mode}: {verdict}")
                    report[mode].setdefault("_verdict", {})[lname] = verdict
                P("")

    # cohort-confounded reference ladder (prereg C3)
    P("#" * 110)
    P("# C3: cohort-CONFOUNDED full-39 ladder (reported for comparability with E142 only)")
    P("#" * 110)
    P(f"{'N':>5} {'>4 f1 M1':>16} {'>4 f1 M2':>16} {'n_seq present':>14} {'n>4 in M2':>10}")
    for N in [1, 2, 4, 8, 16, 24, 32, 0]:
        r = []
        for rule in ("M1", "M2"):
            vv = []
            for sd in SEEDS:
                cd = load(f"r2_N{N}_s{sd}") or (load("r2_N0_s0") if N == 0 else None)
                if cd is None:
                    continue
                vals = [seq_value(m, ">4", "f1", rule) for m in cd["per_seq"].values()]
                vals = [v for v in vals if v is not None]
                if vals:
                    vv.append(float(np.mean(vals)))
            r.append(f"{np.mean(vv):.4f}+-{statistics.pstdev(vv) if len(vv)>1 else 0:.4f}" if vv else "nan")
        cd0 = load(f"r2_N{N}_s0")
        npres = len(cd0["per_seq"]) if cd0 else 0
        n4 = sum(1 for m in (cd0["per_seq"].values() if cd0 else [])
                 if seq_value(m, ">4", "f1", "M2") is not None)
        P(f"{('all' if N == 0 else N):>5} {r[0]:>16} {r[1]:>16} {npres:>14} {n4:>10}")
    P("")

    txt = "\n".join(L)
    (OUT / "results" / "AGG_E147.txt").write_text(txt)
    (OUT / "results" / "ladder_e147.json").write_text(json.dumps(
        {"cohorts": {k: v for k, v in cohorts.items()},
         "T_per_seq": T, "sat_eps": SAT_EPS, "curves_M1_gt4_f1": report}, indent=1))
    print(txt)


if __name__ == "__main__":
    main()
