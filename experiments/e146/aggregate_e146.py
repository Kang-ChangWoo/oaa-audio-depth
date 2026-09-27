#!/usr/bin/env python3
"""E146 aggregator: PREREG_E146 gate G1 + the UP/DOWN/TIE/TRADE verdict, constants hardcoded.

Reads /root/storage/e146_code/collect_e146/<run>__<mae|own>.json (e114_eval.py output) and
integrity_<run>.json, then applies PREREG_E146 sections 4, 6, 7 and 8 mechanically. Every threshold
below is copied from the prereg and must not be edited after results exist.

Arms: ctrl (G1 reproduction anchor) and invfreq (the binding arm). `bin4` is NOT re-run -- it is
`--cue bin4 --loss-weight none`, i.e. binaural 4-channel AUDIO input with no distance weight, so the
radial bin move cannot touch it (PREREG s1.5). Its E145 row is carried as context only.

Includes the n != 3 GATEKEEPER: the pre-registered decision rule is untouched, but a comparison
whose either side is not a full 3-seed cell is WITHHELD rather than decided (a population sd makes
sigma = 0 for a single seed, which would make every difference look decisive).
"""
import json
import math
import os
import statistics as st
import sys

COLLECT = "/root/storage/e146_code/collect_e146"
ARMS = ["ctrl", "invfreq"]
SEEDS = [0, 1, 2]
BINS = ["<0.5", "0.5-1.5", "1.5-4", ">4"]
EPOCHS = 80

# ---- PREREG_E146 s6/s8: the E145 reference, 80 epochs, 3 seeds, (mean, pstdev) ----
E145 = {
    "ctrl": {
        "mae": {"overall": (.5517, .0071), "<0.5": (.6895, .0130), "0.5-1.5": (.6604, .0104),
                "1.5-4": (.5982, .0075), ">4": (.0855, .0037)},
        "own": {"overall": (.5517, .0071), "<0.5": (.6895, .0130), "0.5-1.5": (.6604, .0104),
                "1.5-4": (.5982, .0075), ">4": (.0855, .0037)}},
    "invfreq": {
        "mae": {"overall": (.5532, .0045), "<0.5": (.6580, .0311), "0.5-1.5": (.6520, .0087),
                "1.5-4": (.5950, .0032), ">4": (.1515, .0089)},
        "own": {"overall": (.5406, .0028), "<0.5": (.6846, .0502), "0.5-1.5": (.6387, .0040),
                "1.5-4": (.5720, .0039), ">4": (.2119, .0053)}},
}
# E144 ctrl, used only to show the G1 window is satisfiable by the replication that actually happened
E144_CTRL = {"overall": (.5561, .0054), ">4": (.0883, .0041)}
# context rows, NOT arms
E145_BIN4 = {"overall": (.5759, .0109), ">4": (.1036, .0035)}
E144_INVFREQ_MAE = {"overall": (.5519, .0056), ">4": (.1403, .0135)}
HARD_REFIT_CONTEXT = (.6132, .0004)

PRIMARY_SEL = "mae"        # PREREG s7.1: best.pth -- the only selection rule fixed across E144/5/6
SECONDARY_SEL = "own"      # PREREG s7.2: best_own.pth -- its own criterion moved too
COST_KEYS = ["overall", "<0.5", "0.5-1.5", "1.5-4"]
N_SEEDS_REQUIRED = 3


def pooled_sigma(sa, sb):
    """PREREG_E144 addendum A3, reused verbatim across E144 / E145 / E146."""
    return math.sqrt((sa * sa + sb * sb) / 2.0)


def decide(m146, s146, n146, m145, s145):
    """PREREG_E146 s7.3, UNCHANGED, plus the n != 3 gatekeeper.

    delta = E146 mean - E145 mean;  |delta| < pooled sigma => TIE, else UP / DOWN.
    """
    if m146 is None:
        return "MISSING", None, None
    d = m146 - m145
    p = pooled_sigma(s146, s145)
    if n146 != N_SEEDS_REQUIRED:
        return f"WITHHELD_N(n={n146})", d, p
    if abs(d) < p:
        return "TIE", d, p
    return ("UP" if d > 0 else "DOWN"), d, p


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        # sigma is UNDEFINED for one sample. 0.0 keeps the JSON shape; `n` is what decide() gates on.
        return vals[0], 0.0, 1
    return st.mean(vals), st.pstdev(vals), len(vals)


def load_cell(run, tag):
    p = os.path.join(COLLECT, f"{run}__{tag}.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    agg = j["agg"]
    out = {"n_sequences": j.get("n_sequences"), "note": j.get("e146_note", j.get("e145_note", ""))}
    for k in ["overall"] + BINS:
        a = agg.get(k)
        out[k] = None if not a else {s: a[s] for s in ("precision", "recall", "f1")}
    return out


def load_traindone(run):
    p = os.path.join(COLLECT, f"traindone_{run}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    cells, integ, td = {}, {}, {}
    for arm in ARMS:
        for s in SEEDS:
            run = f"{arm}_s{s}"
            for tag in ("mae", "own"):
                cells[(run, tag)] = load_cell(run, tag)
            ip = os.path.join(COLLECT, f"integrity_{run}.json")
            integ[run] = json.load(open(ip)) if os.path.exists(ip) else None
            td[run] = load_traindone(run)

    missing = [f"{r}__{t}" for (r, t), v in cells.items() if v is None]
    report = {"missing_cells": missing, "prereg": "PREREG_E146.md",
              "eval_md5": "5e28675599c7872e34258a1038f503c8", "epochs": EPOCHS}

    # ---------- PREREG s4: integrity I2 / I3 ----------
    integrity = {"pass": True, "detail": {}}
    for arm in ARMS:
        expect = "IDENTICAL" if arm == "ctrl" else "DIFFERENT"
        for s in SEEDS:
            run = f"{arm}_s{s}"
            got = (integ[run] or {}).get("identical", "MISSING")
            ok = got == expect
            integrity["detail"][run] = {"expect": expect, "got": got, "ok": ok}
            if not ok:
                integrity["pass"] = False
    report["integrity_s4"] = integrity

    # ---------- per-arm means ----------
    summ = {}
    for arm in ARMS:
        summ[arm] = {}
        for tag in ("mae", "own"):
            d = {}
            for k in ["overall"] + BINS:
                per_seed = []
                for s in SEEDS:
                    c = cells[(f"{arm}_s{s}", tag)]
                    per_seed.append(None if not c or not c[k] else c[k]["f1"])
                m, sd, n = ms(per_seed)
                d[k] = {"per_seed": per_seed, "mean": m, "std": sd, "n": n}
            d["best_ep"] = [
                (td[f"{arm}_s{s}"] or {}).get("best_ep_obj" if tag == "own" else "best_ep_mae")
                for s in SEEDS
            ]
            summ[arm][tag] = d
    report["summary"] = summ

    # ---------- PREREG s6: gate G1, ctrl must reproduce E145 ----------
    g1 = {}
    for k in ("overall", ">4"):
        ref_m, ref_s = E145["ctrl"]["mae"][k]
        e144_m, e144_s = E144_CTRL[k]
        p = pooled_sigma(ref_s, e144_s)                      # PREREG s6.1
        lo, hi = ref_m - p, ref_m + p
        cell = summ["ctrl"]["mae"][k]
        got, n = cell["mean"], cell["n"]
        g1[k] = {"ref_e145": ref_m, "pooled_sigma": round(p, 4),
                 "window": [round(lo, 4), round(hi, 4)], "got": None if got is None else round(got, 4),
                 "n": n, "e144_ctrl": e144_m,
                 "e144_inside_window": lo <= e144_m <= hi,
                 "delta_vs_e145": None if got is None else round(got - ref_m, 4),
                 "pass": got is not None and n == N_SEEDS_REQUIRED and lo <= got <= hi}
    g1_pass = all(v["pass"] for v in g1.values())
    report["G1_s6"] = {"detail": g1, "pass": g1_pass,
                       "reading": "PREREG s6.2: this window is wide (single-seed sigma on a 3-seed "
                                  "mean), so a PASS is WEAK evidence of reproduction while a FAIL is "
                                  "STRONG evidence that something moved. A pass does NOT prove the "
                                  "n1->n4 node move (PREREG s3.1) was harmless."}

    # ---------- PREREG s7: primary + secondary verdict on invfreq >4 m ----------
    verdicts = {}
    for sel, label in ((PRIMARY_SEL, "primary"), (SECONDARY_SEL, "secondary")):
        cell = summ["invfreq"][sel][">4"]
        rm, rs = E145["invfreq"][sel][">4"]
        cls, d, p = decide(cell["mean"], cell["std"], cell["n"], rm, rs)
        verdicts[label] = {"selection": "best.pth" if sel == "mae" else "best_own.pth",
                           "metric": ">4", "e145": rm, "e145_sigma": rs,
                           "e146": None if cell["mean"] is None else round(cell["mean"], 4),
                           "e146_sigma": None if cell["std"] is None else round(cell["std"], 4),
                           "n": cell["n"], "delta": None if d is None else round(d, 4),
                           "pooled_sigma": None if p is None else round(p, 4), "class": cls}
    report["verdict_s7"] = verdicts

    # ---------- PREREG s8: the cost sheet, same rule, both selections ----------
    cost = {}
    for sel in (PRIMARY_SEL, SECONDARY_SEL):
        cost[sel] = {}
        for k in COST_KEYS:
            cell = summ["invfreq"][sel][k]
            rm, rs = E145["invfreq"][sel][k]
            cls, d, p = decide(cell["mean"], cell["std"], cell["n"], rm, rs)
            cost[sel][k] = {"e145": rm, "e146": None if cell["mean"] is None else round(cell["mean"], 4),
                            "delta": None if d is None else round(d, 4),
                            "pooled_sigma": None if p is None else round(p, 4),
                            "n": cell["n"], "class": cls}
    report["cost_s8"] = cost

    # ---------- PREREG s8: landing rule ----------
    far = verdicts["primary"]["class"]
    overall = cost[PRIMARY_SEL]["overall"]["class"]
    if missing:
        landing, note = "INCOMPLETE", f"missing cells: {missing}"
    elif far.startswith("WITHHELD") or overall.startswith("WITHHELD"):
        landing, note = "WITHHELD_N", "a cell does not have 3 seeds; PREREG s9 gatekeeper"
    elif not g1_pass:
        landing, note = ("VOID (gate G1 failed)",
                         "PREREG s6.2 + s3.1: G1 FAIL cannot separate 'radial bins' from "
                         "'n1 -> n4 node move'. Next step is reproducing ctrl_s0 on n1.")
    elif far == "UP" and overall in ("UP", "TIE"):
        landing, note = "UP", "clean far-range gain: >4 m UP with overall not degraded"
    elif far == "UP" and overall == "DOWN":
        landing, note = ("TRADE",
                         f"far bought at a price: >4 m {verdicts['primary']['delta']:+.4f}, "
                         f"overall {cost[PRIMARY_SEL]['overall']['delta']:+.4f}")
    elif far == "TIE":
        landing, note = ("TIE",
                         "the mass identity (PREREG s1.4) dominates: invfreq always gives each bin "
                         "25 % of the loss mass, so redefining WHICH pixels are far did not move "
                         "the far-range score. The treatment/measurement distance mismatch was not "
                         "a material confound on this axis. A negative result is a result.")
    else:
        landing, note = "DOWN", "radial bins made the far range worse"
    report["LANDING"] = landing
    report["landing_note"] = note

    out = sys.argv[1] if len(sys.argv) > 1 else "/root/local1/changwoo/e146/results/e146_agg.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)

    # ---------- human-readable ----------
    print("=" * 78)
    print("E146 -- radial loss bins. PREREG_E146.md")
    print("=" * 78)
    if missing:
        print(f"MISSING CELLS: {missing}\n")
    print("## integrity (PREREG s4)")
    for r, v in report["integrity_s4"]["detail"].items():
        print(f"  {r:<12s} expect {v['expect']:<9s} got {v['got']:<9s} {'ok' if v['ok'] else 'FAIL'}")
    print(f"  => {'PASS' if report['integrity_s4']['pass'] else 'FAIL'}\n")

    print("## gate G1 -- ctrl reproduces E145 (PREREG s6)")
    for k, v in g1.items():
        print(f"  {k:>8s}: E145 {v['ref_e145']:.4f}  window [{v['window'][0]:.4f}, {v['window'][1]:.4f}]"
              f"  got {v['got']}  n={v['n']}  d={v['delta_vs_e145']}  "
              f"{'PASS' if v['pass'] else 'FAIL'}   (E144 ctrl {v['e144_ctrl']:.4f} inside: "
              f"{v['e144_inside_window']})")
    print(f"  => G1 {'PASS' if g1_pass else 'FAIL'}")
    print(f"  {report['G1_s6']['reading']}\n")

    print("## arms (F1, mean +- population sd over 3 seeds)")
    hdr = f"{'arm/sel':<16s}" + "".join(f"{k:>12s}" for k in ["overall"] + BINS) + "   best_ep"
    print(hdr)
    for arm in ARMS:
        for sel in ("mae", "own"):
            row = summ[arm][sel]
            cellsf = "".join(
                "     n/a    " if row[k]["mean"] is None else f"{row[k]['mean']:>7.4f}±{row[k]['std']:.4f}"
                for k in ["overall"] + BINS)
            print(f"{arm + '/' + sel:<16s}{cellsf}   {row['best_ep']}")
    print(f"{'bin4 (E145 ctx)':<16s}{E145_BIN4['overall'][0]:>7.4f}±{E145_BIN4['overall'][1]:.4f}"
          f"{'':>36s}{E145_BIN4['>4'][0]:>7.4f}±{E145_BIN4['>4'][1]:.4f}   not re-run (PREREG s1.5)")
    print(f"{'hard refit (ctx)':<16s}{HARD_REFIT_CONTEXT[0]:>7.4f}±{HARD_REFIT_CONTEXT[1]:.4f}"
          f"   context row, NOT an arm\n")

    print("## verdict on invfreq >4 m (PREREG s7)")
    for label in ("primary", "secondary"):
        v = verdicts[label]
        print(f"  {label:<10s} ({v['selection']:<13s}) E145 {v['e145']:.4f}±{v['e145_sigma']:.4f}"
              f" -> E146 {v['e146']}±{v['e146_sigma']}  d={v['delta']}  pooled_s={v['pooled_sigma']}"
              f"  => {v['class']}")
    print("  (secondary's own selection criterion moved to radial too -- PREREG s7.2)\n")

    print("## cost sheet (PREREG s8) -- primary selection, best.pth")
    for k in COST_KEYS:
        c = cost[PRIMARY_SEL][k]
        print(f"  {k:>9s}: E145 {c['e145']:.4f} -> E146 {c['e146']}  d={c['delta']}"
              f"  pooled_s={c['pooled_sigma']}  => {c['class']}")
    print("  NOTE `<0.5` had sigma +-.0311/.0502 in E145 -- quote it only with its sigma (PREREG s8).\n")

    print("=" * 78)
    print(f"## E146 LANDING: {landing}")
    print(f"   {note}")
    print("=" * 78)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
