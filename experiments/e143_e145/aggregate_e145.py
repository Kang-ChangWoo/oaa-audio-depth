#!/usr/bin/env python3
"""E145 aggregator: PREREG_E145 gate G1 + the HOLD/FLIP/TIE verdict, constants hardcoded.

Reads /root/storage/e145_code/collect_e145/<run>__<mae|own>.json (e114_eval.py output) and
integrity_<run>.json, then applies PREREG_E145 sections 4, 5, 6 and 7 mechanically. Every
threshold below is copied from the prereg and must not be edited after results exist.
"""
import json
import math
import os
import statistics as st
import sys

COLLECT = "/root/storage/e145_code/collect_e145"
ARMS = ["bin4", "ctrl", "invfreq"]
SEEDS = [0, 1, 2]
BINS = ["<0.5", "0.5-1.5", "1.5-4", ">4"]
EPOCHS = 80
UNCONV_EP = EPOCHS - 5          # PREREG s7: best_ep_obj >= 75 flags UNCONVERGED_OWN

# ---- PREREG_E145 section 2: E144 reference, 80 epochs, 3 seeds ----
E144 = {
    "bin4":    {"overall": (.5820, .0049), ">4": (.1127, .0098), "best_ep": [41, 37, 36]},
    "ctrl":    {"overall": (.5561, .0054), ">4": (.0883, .0041), "best_ep": [28, 33, 35]},
    "invfreq": {"overall": (.5519, .0056), ">4": (.1403, .0135), "best_ep": [69, 63, 75]},
}
# ---- PREREG_E145 section 6: E144's adjacent-pair classification ----
E144_CLASS = {
    ("overall", "bin4", "ctrl"): "GT",
    ("overall", "ctrl", "invfreq"): "TIE",
    (">4", "invfreq", "bin4"): "GT",
    (">4", "bin4", "ctrl"): "GT",
}
ORDER_O = ["bin4", "ctrl", "invfreq"]      # overall F1
ORDER_F = ["invfreq", "bin4", "ctrl"]      # >4 m F1
HARD_REFIT_CONTEXT = (.6132, .0004)        # context row, NOT an arm


def pooled_sigma(sa, sb):
    """PREREG_E144 addendum A3, reused verbatim."""
    return math.sqrt((sa * sa + sb * sb) / 2.0)


def classify(ma, sa, mb, sb):
    d = ma - mb
    p = pooled_sigma(sa, sb)
    if abs(d) < p:
        return "TIE", d, p
    return ("GT" if d > 0 else "LT"), d, p


def load_cell(run, tag):
    p = os.path.join(COLLECT, f"{run}__{tag}.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    agg = j["agg"]
    out = {"n_sequences": j.get("n_sequences"), "note": j.get("e145_note", "")}
    for k in ["overall"] + BINS:
        a = agg.get(k)
        out[k] = None if not a else {s: a[s] for s in ("precision", "recall", "f1")}
    return out


def load_traindone(run):
    """train_done.json lives on the node the run trained on; the collector copies it next to
    the eval JSONs so the aggregator can read best_ep under both criteria."""
    p = os.path.join(COLLECT, f"traindone_{run}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return st.mean(vals), st.pstdev(vals), len(vals)


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
    report = {"missing_cells": missing}

    # ---------- PREREG s4: integrity ----------
    integrity = {"pass": True, "detail": {}}
    for arm in ARMS:
        expect = "IDENTICAL" if arm in ("ctrl", "bin4") else "ANY"
        for s in SEEDS:
            run = f"{arm}_s{s}"
            got = (integ[run] or {}).get("identical", "MISSING")
            ok = True if expect == "ANY" else (got == expect)
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

    # ---------- PREREG s5: gate G1 ----------
    g1a = {}
    for arm in ARMS:
        g1a[arm] = {}
        for k in ("overall", ">4"):
            ref, sig = E144[arm][k]
            got = summ[arm]["mae"][k]["mean"]
            lo, hi = ref - sig, ref + sig
            g1a[arm][k] = {"e144": ref, "tol_sigma": sig, "interval": [round(lo, 4), round(hi, 4)],
                           "e145_best": got,
                           "delta": None if got is None else round(got - ref, 4),
                           "pass": got is not None and lo <= got <= hi}
    g1b = {}
    for key, e144cls in E144_CLASS.items():
        k, a, b = key
        ma, sa = summ[a]["mae"][k]["mean"], summ[a]["mae"][k]["std"]
        mb, sb = summ[b]["mae"][k]["mean"], summ[b]["mae"][k]["std"]
        if None in (ma, mb):
            g1b[f"{k}:{a}-{b}"] = {"pass": False, "reason": "missing"}
            continue
        cls, d, p = classify(ma, sa, mb, sb)
        g1b[f"{k}:{a}-{b}"] = {"e144_class": e144cls, "e145_best_class": cls,
                               "delta": round(d, 4), "pooled_sigma": round(p, 4),
                               "pass": cls != "LT"}
    invfreq_g1a = all(g1a["invfreq"][k]["pass"] for k in ("overall", ">4"))
    g1b_pass = all(v["pass"] for v in g1b.values())
    others_g1a = all(g1a[a][k]["pass"] for a in ("ctrl", "bin4") for k in ("overall", ">4"))
    if not invfreq_g1a or not g1b_pass:
        g1_state, confidence = "VOID", "none"
    elif others_g1a:
        g1_state, confidence = "PASS", "사실"
    else:
        g1_state, confidence = "PASS_PARTIAL", "추론"
    report["gate_G1"] = {"G1a": g1a, "G1b": g1b, "state": g1_state,
                         "verdict_confidence": confidence,
                         "binding_rule": "invfreq G1a + G1b are binding (PREREG s5)"}

    # ---------- PREREG s7: invfreq convergence flags ----------
    flags = {}
    for s in SEEDS:
        run = f"invfreq_s{s}"
        j = td[run] or {}
        bep = j.get("best_ep_obj")
        hist = j.get("hist") or []
        last5 = None
        if hist and all("val_obj_m" in h for h in hist):
            bo = min(h["val_obj_m"] for h in hist)
            last5 = any(h["val_obj_m"] <= bo + 1e-12 for h in hist[-5:])
        flagged = (bep is not None and bep >= UNCONV_EP) or bool(last5)
        flags[run] = {"best_ep_obj": bep, "last5_improved": last5, "UNCONVERGED_OWN": flagged}
    n_flag = sum(1 for v in flags.values() if v["UNCONVERGED_OWN"])
    report["invfreq_convergence_s7"] = {"per_seed": flags, "n_flagged": n_flag,
                                        "lower_bound_reading": n_flag >= 2}

    # ---------- PREREG s6: verdict on best_own ----------
    pairs = {}
    for k, order in (("overall", ORDER_O), (">4", ORDER_F)):
        for a, b in zip(order, order[1:]):
            ma, sa = summ[a]["own"][k]["mean"], summ[a]["own"][k]["std"]
            mb, sb = summ[b]["own"][k]["mean"], summ[b]["own"][k]["std"]
            if None in (ma, mb):
                pairs[f"{k}:{a}-{b}"] = {"class": "MISSING"}
                continue
            cls, d, p = classify(ma, sa, mb, sb)
            pairs[f"{k}:{a}-{b}"] = {"e144_class": E144_CLASS[(k, a, b)], "e145_own_class": cls,
                                     "delta": round(d, 4), "pooled_sigma": round(p, 4)}
    any_flip = any(v.get("e145_own_class") == "LT" for v in pairs.values())
    any_degrade = any(v.get("e144_class") == "GT" and v.get("e145_own_class") == "TIE"
                      for v in pairs.values())
    if any(v.get("class") == "MISSING" for v in pairs.values()):
        verdict = "INCOMPLETE"
    elif any_flip:
        verdict = "FLIP"
    elif any_degrade:
        verdict = "TIE"
    else:
        verdict = "HOLD"

    # PREREG s7 asymmetry: a FLIP driven by invfreq rising is confounded with unspent budget
    note = ""
    if verdict == "FLIP" and report["invfreq_convergence_s7"]["lower_bound_reading"]:
        rose = False
        for k in ("overall", ">4"):
            mo, mm = summ["invfreq"]["own"][k]["mean"], summ["invfreq"]["mae"][k]["mean"]
            if None not in (mo, mm) and mo > mm:
                rose = True
        if rose:
            verdict = "FLIP [confounded with unspent budget]"
            note = ("PREREG s7: >=2 invfreq seeds are UNCONVERGED_OWN and invfreq rose under "
                    "own-objective selection -> queue E146 (invfreq only, 160 epochs, cosine 160, "
                    "3 seeds, same two selections).")
    if verdict == "HOLD" and report["invfreq_convergence_s7"]["lower_bound_reading"]:
        note = ("PREREG s7: invfreq is a lower bound here and still loses -> HOLD is the STRONG "
                "reading; more budget cannot rescue the ordering by this route.")
    if g1_state == "VOID":
        verdict, note = "VOID (gate G1 failed)", "PREREG s5: no HOLD/FLIP/TIE may be issued."

    report["pairs_own"] = pairs
    report["VERDICT"] = verdict
    report["verdict_note"] = note
    report["delta_own_minus_mae"] = {
        a: {k: (None if None in (summ[a]["own"][k]["mean"], summ[a]["mae"][k]["mean"])
                else round(summ[a]["own"][k]["mean"] - summ[a]["mae"][k]["mean"], 4))
            for k in ["overall"] + BINS} for a in ARMS}
    report["context_row_not_an_arm"] = {"hard_refit_overall_f1": HARD_REFIT_CONTEXT}
    report["eval_budget"] = "FRAC=0.25 top-25% (E123 standing rule); >4m numbers are budget-matched"

    out = sys.argv[1] if len(sys.argv) > 1 else "/root/local1/changwoo/e145/results/e145_agg.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)

    print("=" * 78)
    print("E145 VERDICT:", verdict)
    if note:
        print("  ", note)
    print("gate G1:", g1_state, "| confidence:", confidence)
    print("integrity (PREREG s4) pass:", integrity["pass"])
    if missing:
        print("MISSING CELLS:", missing)
    print("-" * 78)
    for k in ("overall", ">4"):
        print(f"[{k}]")
        for arm in ARMS:
            m = summ[arm]["mae"][k]; o = summ[arm]["own"][k]
            f = lambda x: "  n/a " if x["mean"] is None else f"{x['mean']:.4f}±{x['std']:.4f}"
            print(f"  {arm:8s} best={f(m)}  best_own={f(o)}   e144={E144[arm][k][0]:.4f}"
                  f"±{E144[arm][k][1]:.4f}")
    print("-" * 78)
    for kk, v in pairs.items():
        print(f"  pair {kk:24s} e144={v.get('e144_class'):4s} -> own={v.get('e145_own_class')}"
              f"  d={v.get('delta')} pooled_s={v.get('pooled_sigma')}")
    print("=" * 78)
    print("wrote", out)


if __name__ == "__main__":
    main()
