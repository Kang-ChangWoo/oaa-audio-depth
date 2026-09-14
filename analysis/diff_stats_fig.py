"""Figure + table for analysis/diff_stats.py output (common-mode collapse diagnostic).

  python3 analysis/diff_stats_fig.py            # -> 0820_report/figs/fig12_commonmode.png, table to stdout
"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

MAE = {}
for f, suf in (("comparison_0820/compare.json", ""), ("comparison_0820/mp3d_eval/compare.json", "")):
    if os.path.exists(f):
        MAE |= {k: v["MAE"] for k, v in json.load(open(f)).items()}

recs = {}
for f in ("comparison_0820/diff_stats.json", "comparison_0820/diff_stats_rep.json"):
    if os.path.exists(f):
        recs |= json.load(open(f))
if not recs:
    sys.exit("no diff_stats json yet")

rows = []
for run, r in recs.items():
    ds = "MP3D" if r["dataset"] == "data_mp3d" else "Replica"
    pts = sorted(((r["live_mics"][t], s) for t, s in r["stats"].items()), reverse=True)
    rows.append((ds, run, r["afm_stem"], MAE.get(run, float("nan")), pts))
rows.sort(key=lambda x: (x[0], x[3]))

print(f"{'dataset':8s} {'run':32s} {'stem':7s} {'MAE':>7s} " +
      " ".join(f"{m}mic dratio/cos" for m, _ in rows[0][4]))
for ds, run, stem, mae, pts in rows:
    cells = "  ".join(f"{s['dratio']:.3f}/{s['cos']:.3f}" for _, s in pts)
    print(f"{ds:8s} {run:32s} {stem:7s} {mae:7.4f}  {cells}")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
for ax, key, ylab, title in (
        (axes[0], "dratio", r"$\|D_i\|/\|S_i\|$", "mic-specific share of the AFM feature"),
        (axes[1], "cos", r"mean pairwise $\cos(S_i,S_j)$", "how alike the observations look")):
    for ds, run, stem, mae, pts in rows:
        col = "crimson" if ds == "MP3D" else "steelblue"
        ls = "-" if stem != "linear" else "--"
        ax.plot([m for m, _ in pts], [s[key] for _, s in pts], ls, color=col, marker="o", ms=5,
                lw=2.0, alpha=.85, label=f"{ds} {run.replace('0820_','')} ({mae:.3f})")
    ax.invert_xaxis(); ax.set_xlabel("live mics"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.grid(alpha=.3); ax.spines[["top", "right"]].set_visible(False)
axes[1].axhline(1.0, color="k", lw=.8, ls=":")
axes[1].legend(fontsize=7, frameon=False, loc="lower left")
fig.suptitle("Common-mode collapse diagnostic — MAE in brackets · solid: convstem · dashed: linear patch embed")
fig.tight_layout()
os.makedirs("0820_report/figs", exist_ok=True)
fig.savefig("0820_report/figs/fig12_commonmode.png", dpi=120)
print("\n[saved] 0820_report/figs/fig12_commonmode.png")
