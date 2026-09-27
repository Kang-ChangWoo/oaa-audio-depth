#!/usr/bin/env python3
"""Add an n != 3 GATEKEEPER to the E144 / E145 aggregators.

The bug: both aggregators compute a POPULATION standard deviation, so a cell that lost seeds and has
only one surviving run reports sigma = 0.0. The pre-registered decision rule is "|delta| < pooled
sigma => TIE", so sigma = 0 makes every difference, however tiny, a decisive GT/LT.

The DECISION RULE IS PRE-REGISTERED AND IS NOT TOUCHED. This patch only adds a gate: when either
side of a comparison has n != 3 seeds, the comparison is WITHHELD instead of decided.

Every replacement is asserted to match exactly once. Backups: <file>.bak-pre-ngate-20260927
"""
import hashlib, os, shutil, sys

E145_MS_OLD = '''def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return st.mean(vals), st.pstdev(vals), len(vals)
'''

E145_MS_NEW = '''def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        # sigma is UNDEFINED for one sample. 0.0 is kept only so the JSON stays the same shape --
        # `n` is what callers must gate on, see classify()'s WITHHELD_N branch.
        return vals[0], 0.0, 1
    return st.mean(vals), st.pstdev(vals), len(vals)
'''

E145_CLS_OLD = '''def classify(ma, sa, mb, sb):
    d = ma - mb
    p = pooled_sigma(sa, sb)
    if abs(d) < p:
        return "TIE", d, p
    return ("GT" if d > 0 else "LT"), d, p
'''

E145_CLS_NEW = '''N_SEEDS_REQUIRED = 3


def classify(ma, sa, mb, sb, na=N_SEEDS_REQUIRED, nb=N_SEEDS_REQUIRED):
    """PREREG decision rule, UNCHANGED, plus an n != 3 gatekeeper.

    `ms()` reports a POPULATION sd, so a cell with one surviving seed has sigma = 0.0 and the rule
    "|delta| < pooled sigma => TIE" would call every difference decisive. The rule is pre-registered
    and is not modified; instead a comparison whose either side is not a full 3-seed cell is
    WITHHELD. A withheld comparison carries no verdict and must be reported as `n=k WITHHELD`.
    """
    d = ma - mb
    p = pooled_sigma(sa, sb)
    if na != N_SEEDS_REQUIRED or nb != N_SEEDS_REQUIRED:
        return f"WITHHELD_N(na={na},nb={nb})", d, p
    if abs(d) < p:
        return "TIE", d, p
    return ("GT" if d > 0 else "LT"), d, p
'''

# the two call sites in aggregate_e145.py -- pass the seed counts through
E145_CALL1_OLD = '''        ma, sa = summ[a]["mae"][k]["mean"], summ[a]["mae"][k]["std"]
        mb, sb = summ[b]["mae"][k]["mean"], summ[b]["mae"][k]["std"]
        if None in (ma, mb):
            g1b[f"{k}:{a}-{b}"] = {"pass": False, "reason": "missing"}
            continue
        cls, d, p = classify(ma, sa, mb, sb)'''

E145_CALL1_NEW = '''        ma, sa = summ[a]["mae"][k]["mean"], summ[a]["mae"][k]["std"]
        mb, sb = summ[b]["mae"][k]["mean"], summ[b]["mae"][k]["std"]
        if None in (ma, mb):
            g1b[f"{k}:{a}-{b}"] = {"pass": False, "reason": "missing"}
            continue
        cls, d, p = classify(ma, sa, mb, sb, summ[a]["mae"][k]["n"], summ[b]["mae"][k]["n"])'''

E145_CALL2_OLD = '''            ma, sa = summ[a]["own"][k]["mean"], summ[a]["own"][k]["std"]
            mb, sb = summ[b]["own"][k]["mean"], summ[b]["own"][k]["std"]
            if None in (ma, mb):
                pairs[f"{k}:{a}-{b}"] = {"class": "MISSING"}
                continue
            cls, d, p = classify(ma, sa, mb, sb)'''

E145_CALL2_NEW = '''            ma, sa = summ[a]["own"][k]["mean"], summ[a]["own"][k]["std"]
            mb, sb = summ[b]["own"][k]["mean"], summ[b]["own"][k]["std"]
            if None in (ma, mb):
                pairs[f"{k}:{a}-{b}"] = {"class": "MISSING"}
                continue
            cls, d, p = classify(ma, sa, mb, sb,
                                 summ[a]["own"][k]["n"], summ[b]["own"][k]["n"])'''

E144_SD_OLD = '''def sd(xs):
    return float(np.std(np.asarray(xs, dtype=float), ddof=0))   # population sd, as in E143
'''

E144_SD_NEW = '''N_SEEDS_REQUIRED = 3


def sd(xs):
    return float(np.std(np.asarray(xs, dtype=float), ddof=0))   # population sd, as in E143
'''

E144_ORDER_OLD = '''def order_report(label, means, sds, ref_order):
    got = rank(means)
    ties = []
    for a, b in zip(got, got[1:]):
        ps = pooled_sigma(sds[a], sds[b])
        diff = abs(means[a] - means[b])
        if diff < ps:
            ties.append((a, b, diff, ps))
    print(f"{label}: {' > '.join(f'{a} {means[a]:.4f}' for a in got)}")'''

E144_ORDER_NEW = '''def order_report(label, means, sds, ref_order, ns=None):
    """PREREG ordering rule, UNCHANGED, plus an n != 3 gatekeeper.

    `sd()` is a POPULATION sd, so an arm with one surviving seed has sigma = 0.0 and the "|delta| <
    pooled sigma => TIE" test can never fire for it. The rule is pre-registered and is not modified;
    instead, if any arm in this ordering is not a full 3-seed cell the ordering is WITHHELD.
    """
    got = rank(means)
    withheld = [] if ns is None else [a for a in got if ns.get(a) != N_SEEDS_REQUIRED]
    if withheld:
        print(f"{label}: WITHHELD -- arms with n != {N_SEEDS_REQUIRED}: "
              + ", ".join(f"{a} n={ns.get(a)}" for a in withheld))
        return {"order": list(got), "ref_order": list(ref_order), "changed": None,
                "ties": [], "tie_explained": None,
                "WITHHELD_N": {a: ns.get(a) for a in withheld}}
    ties = []
    for a, b in zip(got, got[1:]):
        ps = pooled_sigma(sds[a], sds[b])
        diff = abs(means[a] - means[b])
        if diff < ps:
            ties.append((a, b, diff, ps))
    print(f"{label}: {' > '.join(f'{a} {means[a]:.4f}' for a in got)}")'''

E144_CALLS_OLD = '''        oo = order_report("ORDER_O (overall F1)", means, sds, ORDER_O_40)
        ff = order_report("ORDER_F (>4 m F1)", meansF, sdsF, ORDER_F_40)'''

E144_CALLS_NEW = '''        oo = order_report("ORDER_O (overall F1)", means, sds, ORDER_O_40, nseeds)
        ff = order_report("ORDER_F (>4 m F1)", meansF, sdsF, ORDER_F_40, nseeds)'''

E144_NSEEDS_OLD = '''        means[arm], sds[arm] = float(np.mean(O)), sd(O)
        meansF[arm], sdsF[arm] = float(np.mean(F)), sd(F)'''

E144_NSEEDS_NEW = '''        means[arm], sds[arm] = float(np.mean(O)), sd(O)
        meansF[arm], sdsF[arm] = float(np.mean(F)), sd(F)
        nseeds[arm] = len(O)                      # n != 3 gatekeeper input'''

JOBS = {
    "/root/storage/e145_code/aggregate_e145.py": [
        (E145_MS_OLD, E145_MS_NEW), (E145_CLS_OLD, E145_CLS_NEW),
        (E145_CALL1_OLD, E145_CALL1_NEW), (E145_CALL2_OLD, E145_CALL2_NEW)],
    "/root/storage/e144_code/aggregate_e144.py": [
        (E144_SD_OLD, E144_SD_NEW), (E144_ORDER_OLD, E144_ORDER_NEW),
        (E144_NSEEDS_OLD, E144_NSEEDS_NEW), (E144_CALLS_OLD, E144_CALLS_NEW)],
}

fail = 0
for path, repls in JOBS.items():
    if not os.path.exists(path):
        print(f"[skip] {path} missing")
        continue
    src = open(path).read()
    print(f"\n=== {path}  md5 {hashlib.md5(src.encode()).hexdigest()}")
    if "N_SEEDS_REQUIRED" in src:
        print("[skip] already gated")
        continue
    out = src
    ok = True
    for i, (old, new) in enumerate(repls, 1):
        n = out.count(old)
        if n != 1:
            print(f"[FAIL] replacement {i}: matched {n} times, expected 1")
            ok = False
            break
        out = out.replace(old, new)
        print(f"[ok] replacement {i}: 1 match")
    if not ok:
        fail += 1
        continue
    if path.endswith("aggregate_e144.py"):
        # `nseeds` must exist before the arm loop writes into it
        anchor = "    means, sds, meansF, sdsF, summary = {}, {}, {}, {}, {}\n"
        if out.count(anchor) != 1:
            print(f"[FAIL] nseeds anchor matched {out.count(anchor)} times")
            fail += 1
            continue
        out = out.replace(anchor, anchor + "    nseeds = {}\n", 1)
        print("[ok] declared nseeds")
    shutil.copy2(path, path + ".bak-pre-ngate-20260927")
    open(path, "w").write(out)
    print(f"[ok] wrote {path}  md5 {hashlib.md5(out.encode()).hexdigest()}")

sys.exit(1 if fail else 0)
