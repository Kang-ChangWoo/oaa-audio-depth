#!/usr/bin/env python3
"""E144 aggregation + verdict, applying PREREG_E144.md sec. 3-4 and addendum A2/A3 mechanically.

Every constant below was fixed BEFORE any E144 number existed (see the PREREG addendum, written
at launch time). Do not edit them after the fact -- that is the whole point of the pre-registration.

Inputs, all on shared NFS so one call sees runs from both nodes:
  $E144_COLLECT/<run>.json              -- e114_eval.py output, copied by score_e144.sh
  $E144_COLLECT/<run>.train_done.json   -- trainer history, copied by collect_e144.sh
"""
import json
import math
import os
import sys

import numpy as np

COLLECT = os.environ.get("E144_COLLECT", "/root/storage/e143_code/collect_e144")
OUT_JSON = os.environ.get("E144_AGG_OUT", "")
ARMS = ("ctrl", "bin4", "invfreq")
SEEDS = (0, 1, 2)
BINS = ("<0.5", "0.5-1.5", "1.5-4", ">4")
EPOCHS = 80

# --- pre-registered 40 ep reference rows (PREREG addendum A2, from REPORT_E143.md sec. 3). ---
REF40 = {
    "ctrl":    {"O": .5601, "O_sd": .0053, "F": .0864, "F_sd": .0038,
                "mae": (.3280, .3385, .3421), "best_ep": (29, 31, 30)},
    "bin4":    {"O": .5820, "O_sd": .0087, "F": .1106, "F_sd": .0101,
                "mae": (.3082, .3049, .2940), "best_ep": (33, 33, 31)},
    "invfreq": {"O": .5513, "O_sd": .0097, "F": .1873, "F_sd": .0022,
                "mae": (.4168, .4275, .4184), "best_ep": (39, 39, 39)},
}
ORDER_O_40 = ("bin4", "ctrl", "invfreq")      # overall F1, high -> low
ORDER_F_40 = ("invfreq", "bin4", "ctrl")      # >4 m F1, high -> low
# context row only -- NOT an E144 arm and NOT used in the verdict (addendum A2).
HARD_REFIT = {"O": .6132, "O_sd": .0004, "F": .0915}
CONVERGED_TAIL = 5      # PREREG sec. 4.1: best inside the last 5 epochs => still unconverged
BUDGET_BOUND = 76       # PREREG sec. 4.1: best_ep >= 76 of 80 => 80 is also not enough


def pooled_sigma(sa, sb):
    """PREREG addendum A3, fixed before any E144 number was seen."""
    return math.sqrt((sa * sa + sb * sb) / 2.0)


def sd(xs):
    return float(np.std(np.asarray(xs, dtype=float), ddof=0))   # population sd, as in E143


def pick(d, key):
    """eval json -> f1 for `key`, tolerating {'agg': ...} or flat layouts (same as aggregate_e143)."""
    node = d.get("agg", d)
    v = node.get(key)
    if isinstance(v, dict):
        for k in ("f1_mean", "f1"):
            if k in v:
                return float(v[k])
    raise KeyError(f"cannot find f1 for {key} in {list(node)[:8]}")


def load(run, suffix=".json"):
    p = os.path.join(COLLECT, run + suffix)
    return json.load(open(p)) if os.path.exists(p) else None


def rank(means):
    """arms high -> low by mean."""
    return tuple(sorted(means, key=lambda a: -means[a]))


def order_report(label, means, sds, ref_order):
    got = rank(means)
    ties = []
    for a, b in zip(got, got[1:]):
        ps = pooled_sigma(sds[a], sds[b])
        diff = abs(means[a] - means[b])
        if diff < ps:
            ties.append((a, b, diff, ps))
    print(f"{label}: {' > '.join(f'{a} {means[a]:.4f}' for a in got)}")
    print(f"  40ep order was: {' > '.join(ref_order)}")
    for a, b, diff, ps in ties:
        print(f"  TIE {a} vs {b}: |Δ|={diff:.4f} < pooled σ={ps:.4f}  (not counted as FLIP)")
    tie_pairs = {frozenset((a, b)) for a, b, _, _ in ties}
    # a swap explained solely by tied adjacent pairs is a tie, not a flip (PREREG sec. 3)
    changed = got != tuple(ref_order)
    explained = changed and all(
        frozenset((x, y)) in tie_pairs
        for x, y in zip(ref_order, ref_order[1:])
        if got.index(x) > got.index(y)
    )
    return {"order": list(got), "ref_order": list(ref_order), "changed": changed,
            "ties": [[a, b, diff, ps] for a, b, diff, ps in ties],
            "tie_explained": bool(explained)}


def main():
    rows = {}
    conv = {}
    missing = []
    for arm in ARMS:
        per_seed = []
        for s in SEEDS:
            run = f"{arm}_s{s}"
            d = load(run)
            td = load(run, ".train_done.json")
            if d is None:
                missing.append(run)
                continue
            rec = {"seed": s, "O": pick(d, "overall"), "F": pick(d, ">4"),
                   "bins": {b: pick(d, b) for b in BINS}}
            if td:
                hist = td["hist"]
                best = float(td["best_val_mae_m"])
                bep = min(hist, key=lambda h: h["val_mae_m"])["epoch"]
                last_imp = any(h["val_mae_m"] <= best + 1e-12 for h in hist[-CONVERGED_TAIL:])
                rec.update({"best_val_mae_m": best, "best_ep": bep, "n_ep": len(hist),
                            "last5_improved": bool(last_imp),
                            "budget_bound": bool(bep >= BUDGET_BOUND)})
            per_seed.append(rec)
        if per_seed:
            rows[arm] = per_seed
            conv[arm] = [r for r in per_seed if "best_ep" in r]

    if missing:
        print(f"[pending] {len(missing)} run(s) not scored yet: {' '.join(missing)}")
    if not rows:
        sys.exit("no runs scored yet")

    print("E144 — 80 epoch re-prescription of ctrl / bin4 / invfreq, 3 seeds")
    print("eval: e114_eval.py UNMODIFIED (md5 5e28675599c7872e34258a1038f503c8), FRAC .25, voxel 0.1 m, B_argmax, 39 test seqs")
    print("framing: NOT 'same recipe, longer' -- the cosine horizon doubled too (PREREG sec. 1 / A1)")
    print()

    print("## per-seed (80ep)")
    hdr = f"{'run':<14}{'overall':>9}{'<0.5':>9}{'0.5-1.5':>9}{'1.5-4':>9}{'>4':>9}{'val MAE':>10}{'best ep':>9}{'last5?':>8}"
    print(hdr)
    for arm in ARMS:
        for r in rows.get(arm, []):
            print(f"{arm + '_s' + str(r['seed']):<14}{r['O']:>9.4f}"
                  + "".join(f"{r['bins'][b]:>9.4f}" for b in BINS)
                  + f"{r.get('best_val_mae_m', float('nan')):>10.4f}"
                  + f"{r.get('best_ep', -1):>9}"
                  + f"{'YES' if r.get('last5_improved') else 'no':>8}")
    print()

    means, sds, meansF, sdsF, summary = {}, {}, {}, {}, {}
    print("## arm summary (mean ± population σ over seeds), and 40ep deltas")
    for arm in ARMS:
        rs = rows.get(arm, [])
        if len(rs) < 2:
            print(f"{arm}: only {len(rs)} seed(s) -- excluded from the verdict")
            continue
        O = [r["O"] for r in rs]
        F = [r["F"] for r in rs]
        means[arm], sds[arm] = float(np.mean(O)), sd(O)
        meansF[arm], sdsF[arm] = float(np.mean(F)), sd(F)
        ref = REF40[arm]
        binm = {b: (float(np.mean([r["bins"][b] for r in rs])), sd([r["bins"][b] for r in rs])) for b in BINS}
        maes = [r["best_val_mae_m"] for r in rs if "best_val_mae_m" in r]
        summary[arm] = {
            "n_seeds": len(rs),
            "O": means[arm], "O_sd": sds[arm], "O_40": ref["O"], "dO": means[arm] - ref["O"],
            "F": meansF[arm], "F_sd": sdsF[arm], "F_40": ref["F"], "dF": meansF[arm] - ref["F"],
            "bins": {b: list(binm[b]) for b in BINS},
            "best_val_mae_mean": float(np.mean(maes)) if maes else None,
            "best_val_mae_40_mean": float(np.mean(ref["mae"])),
            "best_ep": [r.get("best_ep") for r in rs],
            "best_ep_40": list(ref["best_ep"]),
            "last5_improved": [r.get("last5_improved") for r in rs],
            "budget_bound": [r.get("budget_bound") for r in rs],
        }
        s = summary[arm]
        print(f"{arm:<9} O {s['O']:.4f} ± {s['O_sd']:.4f}  (40ep {s['O_40']:.4f}, Δ {s['dO']:+.4f})"
              f"   >4m {s['F']:.4f} ± {s['F_sd']:.4f}  (40ep {s['F_40']:.4f}, Δ {s['dF']:+.4f})")
        if s["best_val_mae_mean"] is not None:
            print(f"{'':<9} best val MAE {s['best_val_mae_mean']:.4f} (40ep {s['best_val_mae_40_mean']:.4f}, "
                  f"Δ {s['best_val_mae_mean'] - s['best_val_mae_40_mean']:+.4f})   best_ep {s['best_ep']} "
                  f"(40ep {s['best_ep_40']})  last5_improved {s['last5_improved']}")
        print(f"{'':<9} near-range cost vs 40ep ctrl: "
              + "  ".join(f"{b} {binm[b][0] - 0:.4f}" for b in BINS))
    print()
    print(f"context row (not an arm): hard refit baseline O {HARD_REFIT['O']:.4f} ± {HARD_REFIT['O_sd']:.4f} "
          f"/ >4m {HARD_REFIT['F']:.4f}  [REPORT_E143.md:21]")
    print()

    if len(means) < 3:
        print("## VERDICT: PENDING -- fewer than three arms have >=2 scored seeds")
        verdict = "PENDING"
        oo = ff = None
    else:
        print("## pre-registered ordering test (PREREG sec. 3, pooled σ per addendum A3)")
        oo = order_report("ORDER_O (overall F1)", means, sds, ORDER_O_40)
        ff = order_report("ORDER_F (>4 m F1)", meansF, sdsF, ORDER_F_40)
        print()
        f_top_ok = rank(meansF)[0] == "invfreq"
        hold = (not oo["changed"]) and f_top_ok
        flip = (oo["changed"] and not oo["tie_explained"]) or (not f_top_ok and not ff["tie_explained"])
        verdict = "HOLD" if hold else ("FLIP" if flip else "TIE")
        print(f"## VERDICT: {verdict}")
        if verdict == "HOLD":
            print("  ORDER_O is bin4 > ctrl > invfreq and invfreq is top at >4 m.")
            print("  => the 40ep-based verdicts (E114b / E134 / E135 / E143 / E143b) are NOT voided by the")
            print("     convergence-budget confound.")
        elif verdict == "FLIP":
            print("  at least one pre-registered order changed, and not merely into a tie.")
            print("  => every 40ep-based verdict must be re-examined, specifically: (1) E143b's 'the fusion")
            print("     axis is closed', (2) E135's adoption of bin4 alone, (3) E114b's invfreq-is-worse call.")
            print("     Those three rest on the 40ep table and must be rewritten against the 80ep table.")
        else:
            print("  an order changed only into a tie: not a FLIP, but record that what was a ranking at")
            print("  40ep is a tie at 80ep (PREREG sec. 3, last bullet).")

    print()
    print("## PREREG sec. 4.1 -- did every arm's best epoch sit at the end of the budget?")
    for arm in ARMS:
        for r in conv.get(arm, []):
            flagged = r["best_ep"] >= BUDGET_BOUND or r["last5_improved"]
            print(f"  {arm}_s{r['seed']}: best_ep {r['best_ep']}/{EPOCHS} n_ep {r['n_ep']} "
                  f"last5_improved={'YES' if r['last5_improved'] else 'no'} "
                  f"-> {'STILL UNCONVERGED at 80 (its numbers are unconverged values)' if flagged else 'converged'}")

    print()
    print("## PREREG sec. 5 -- confound this experiment does NOT fix")
    print("  best.pth is selected on quick_val's UNWEIGHTED MAE, which is not the objective of the")
    print("  weighted arms. Independent of the epoch budget; 80 epochs does not fix it.")

    if OUT_JSON:
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        json.dump({"verdict": verdict, "arms": summary, "per_seed": rows,
                   "order_O": oo, "order_F": ff, "pending": missing,
                   "ref40": REF40, "hard_refit": HARD_REFIT,
                   "eval_md5": "5e28675599c7872e34258a1038f503c8", "epochs": EPOCHS},
                  open(OUT_JSON, "w"), indent=2)
        print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
