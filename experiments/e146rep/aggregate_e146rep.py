# E146REP aggregator. Applies PREREG_E146REP section 3 EXACTLY as written, no post-hoc changes.
#   sigma_rep = POPULATION sd of >4 m F1 over the three same-cell observations of ctrl_s2
#               (E145, E146, rep1), compared against sigma_G1 = .0039 (the half-width G1 used).
#   sigma_rep >= sigma_G1 -> GATE_UNDERPOWERED     sigma_rep < sigma_G1 -> DRIFT_REAL
# Every number is read off disk. Writes results/verdict_e146rep.json. No GPU.
import json, statistics as st

E145 = "/root/storage/e145_code/collect_e145"          # shared NFS: holds the n1 runs too
E146 = "/root/storage/e146_code/collect_e146"
REP  = "/root/local1/changwoo/e146rep/eval_results"
SIGMA_G1 = 0.003905                                     # PREREG_E146 s6.1, unrounded (REPORT_E146 s14.2)
GATE_CENTER = 0.0855                                    # the value G1 was actually executed with

def f1(path, bin_=">4"):
    return json.load(open(path))["agg"][bin_]["f1"]

cell = {
    "E145_ctrl_s2": f1(E145 + "/ctrl_s2__mae.json"),
    "E146_ctrl_s2": f1(E146 + "/ctrl_s2__mae.json"),
    "rep1_ctrl_s2": f1(REP  + "/ctrl_s2_rep1__mae.json"),
}
vals = list(cell.values())
sigma_rep_pop  = st.pstdev(vals)
sigma_rep_samp = st.stdev(vals)
verdict = "GATE_UNDERPOWERED" if sigma_rep_pop >= SIGMA_G1 else "DRIFT_REAL"

# --- scale audit: what quantity did G1 actually gate? (the three numbers are NOT commensurable) ---
ctrl145 = [f1(E145 + "/ctrl_s%d__mae.json" % s) for s in (0, 1, 2)]
ctrl146 = [f1(E146 + "/ctrl_s%d__mae.json" % s) for s in (0, 1, 2)]
m145, m146 = sum(ctrl145) / 3, sum(ctrl146) / 3
upper = GATE_CENTER + SIGMA_G1
audit = {
    "gated_quantity": "3-seed MEAN of ctrl >4 m F1 vs window centered on E145 ctrl mean",
    "E145_ctrl_mean": m145, "E146_ctrl_mean": m146,
    "gate_window": [GATE_CENTER - SIGMA_G1, upper],
    "mean_shift_that_was_gated": m146 - GATE_CENTER,
    "mean_shift_in_sigma_G1": (m146 - GATE_CENTER) / SIGMA_G1,
    "overshoot_past_upper_edge": m146 - upper,          # this is the 1.25e-5 quoted in REPORT_E146
    "pairwise_rep_gap_single_runs": cell["rep1_ctrl_s2"] - cell["E146_ctrl_s2"],
    "per_seed_shift_E145_to_E146": {"s%d" % i: ctrl146[i] - ctrl145[i] for i in range(3)},
    "per_seed_shift_in_sigma_rep": {"s%d" % i: (ctrl146[i] - ctrl145[i]) / sigma_rep_pop for i in range(3)},
    "se_of_3seed_mean_from_sigma_rep": sigma_rep_pop / 3 ** 0.5,
    "mean_shift_in_se_of_difference": (m146 - m145) / (sigma_rep_pop / 3 ** 0.5 * 2 ** 0.5),
}

# --- secondary axis: the OTHER gate G1 checked. Was IT underpowered too? ---
ov = {
    "E145_ctrl_s2": f1(E145 + "/ctrl_s2__mae.json", "overall"),
    "E146_ctrl_s2": f1(E146 + "/ctrl_s2__mae.json", "overall"),
    "rep1_ctrl_s2": f1(REP  + "/ctrl_s2_rep1__mae.json", "overall"),
}
SIGMA_G1_OVERALL = 0.006308                             # PREREG_E146 s6.1
sigma_rep_overall = st.pstdev(list(ov.values()))
overall_axis = {
    "cells": ov, "sigma_rep_population": sigma_rep_overall,
    "sigma_G1_overall": SIGMA_G1_OVERALL,
    "ratio_sigma_rep_over_sigma_G1": sigma_rep_overall / SIGMA_G1_OVERALL,
    "note": "G1 PASSED on overall with d=.0019; this ratio says how much margin that pass really had",
}
print("overall axis: sigma_rep %.6f vs sigma_G1 %.6f  ratio %.2f" % (
    sigma_rep_overall, SIGMA_G1_OVERALL, sigma_rep_overall / SIGMA_G1_OVERALL))

out = {
    "prereg": "PREREG_E146REP.md s3", "metric": ">4 m F1, best.pth (mae selection)",
    "cells": cell, "sigma_rep_population": sigma_rep_pop, "sigma_rep_sample": sigma_rep_samp,
    "sigma_G1": SIGMA_G1, "verdict": verdict,
    "verdict_robust_to_ddof": (sigma_rep_pop >= SIGMA_G1) == (sigma_rep_samp >= SIGMA_G1),
    "side_observation_overall_f1_rep1": f1(REP + "/ctrl_s2_rep1__mae.json", "overall"),
    "scale_audit": audit, "overall_axis": overall_axis,
}
json.dump(out, open("/root/local1/changwoo/e146rep/results/verdict_e146rep.json", "w"), indent=1)
for k, v in cell.items():
    print("  %-14s %.6f" % (k, v))
print("sigma_rep(pop)  %.6f   sigma_rep(sample) %.6f   sigma_G1 %.6f" % (sigma_rep_pop, sigma_rep_samp, SIGMA_G1))
print("VERDICT %s   (robust to ddof: %s)" % (verdict, out["verdict_robust_to_ddof"]))
print("scale audit:")
print("  gated mean shift %+.6f = %.4f sigma_G1 ; overshoot past edge %.3e" % (
    audit["mean_shift_that_was_gated"], audit["mean_shift_in_sigma_G1"], audit["overshoot_past_upper_edge"]))
print("  single-run pairwise gap E146 vs rep1 %+.6f" % audit["pairwise_rep_gap_single_runs"])
for s in ("s0", "s1", "s2"):
    print("  %s shift %+.6f = %+.2f sigma_rep" % (s, audit["per_seed_shift_E145_to_E146"][s], audit["per_seed_shift_in_sigma_rep"][s]))
print("  mean shift %+.6f = %.2f x SE_diff" % (m146 - m145, audit["mean_shift_in_se_of_difference"]))
