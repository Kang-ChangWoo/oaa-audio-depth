import glob
import json
import os
import statistics as st

ORDER = ["true", "randyaw", "randyaw2", "randyaw3", "yawblind", "yawroll",
         "swap", "blind", "earblind", "randear", "randear2", "randear3",
         "earflip1", "earflip2", "earflip3", "earleft",
         "spread", "spreadear", "pairbreak", "spreadoff", "spreadoffear",
         "spreadperm", "spreadperm2", "spreadpermear", "spreadpermear2"]
R = "/root/local1/changwoo/poseablate_20261106/results"

rows = {}
for f in sorted(glob.glob(os.path.join(R, "poseablate_rep_*json"))):
    d = json.load(open(f))
    for cell, modes in d.items():
        t = modes.get("true", {}).get("AbsRel")
        if t is None:
            continue
        rows.setdefault(cell, {"true": t})
        for m, rec in modes.items():
            if m == "true" or m.startswith("_"):
                continue
            if rec.get("_identity_noop"):
                continue
            rows[cell][m] = 100.0 * (rec["AbsRel"] - t) / t

for cell in sorted(rows):
    got = [m for m in ORDER if m in rows[cell] and m != "true"]
    print(f"{cell:26s} true {rows[cell]['true']:.5f}  " +
          "  ".join(f"{m} {rows[cell][m]:+.0f}%" for m in got))

print()
print("=== mode means over the cells that have them ===")
groups = {"4obs ctrl": "ctrl_fb", "4obs ILD": "ild_fb", "8obs ctrl": "ctrl_r8", "8obs ILD": "ild_r8",
          "2obs ctrl": "ctrl_r2", "2obs ILD": "ild_r2", "2obs IPD": "ipd_r2",
          "2obs ILDIPD": "ildipd_r2", "2obs ILDMIN": "ildmin_r2"}
for label, key in groups.items():
    cells = [c for c in rows if key in c]
    if not cells:
        continue
    out = []
    for m in ORDER[1:]:
        vals = [rows[c][m] for c in cells if m in rows[c]]
        if vals:
            out.append(f"{m} {st.mean(vals):+.0f}% (n={len(vals)})")
    if out:
        print(f"{label:12s} cells={len(cells)}  " + "  ".join(out))
