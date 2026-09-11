"""Merge the micdrop shards into the study table + figure.

  python3 analysis/micdrop_study/report.py

Reads analysis/micdrop_study/shards/*.json (written by run.sh), writes curves.json, table.md
and curve.png next to them. Missing models are reported and skipped, so this is safe to run
while the sweep is still filling in.
"""
import os, sys, json, glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from analysis.micdrop_study.roster import ROSTER, ECO, GROUP_ORDER, GROUP_LABEL

HERE = "analysis/micdrop_study"
MICS = list(range(8, 0, -1))          # 8..1 live mics  (k = 8 - mics)


def load():
    """-> {run: {mics: MAE}} merged from every shard."""
    out = {}
    for f in glob.glob(f"{HERE}/shards/*.json"):
        for run, rec in json.load(open(f)).items():
            c = rec["curve"]
            out[run] = {8: c["k0"]} | {
                8 - k: c[f"k{k}"]["mean"] for k in range(1, 8) if f"k{k}" in c}
    return out


def main():
    data = load()
    roster = ROSTER + [ECO] if ECO[2] in data else ROSTER
    have = [(lbl, grp, run) for lbl, grp, run in roster if run in data]
    missing = [run for _, _, run in roster if run not in data]

    # ---- table: ONE ranking over every model in the experiment. Training-time mic drop is a
    # column, not a separate section — models trained with and without it are compared directly.
    TRAIN_COL = {"novd": "no", "prior": "no", "vdrop": "yes"}
    ORIGIN = {"novd": "ours", "prior": "prior work", "vdrop": "ours"}
    L = ["# Mic-drop degradation: every model trained on 8 mics, tested on 8-1", "",
         "Replica test (off3), MAE in metres. At inference k of the 8 channels are zeroed and the",
         "pose conditioning stays truthful — the *mic failure* curve, not a smaller-rig retrain.",
         "Each k is the mean over 3 fixed-seed random subsets. `mic-drop` says whether the model saw",
         "dropped mics during TRAINING (no prior-work model does). `ret@1` = MAE(1 mic)/MAE(8 mic):",
         "how much of the 8-mic error survives total rig loss — lower is a flatter, more graceful",
         "curve. Sorted by 1-mic MAE (worst case first-class, since that is what the study is about).",
         ""]
    head = "| model | origin | mic-drop | " + " | ".join(f"{m}mic" for m in MICS) + " | ret@1 |"
    L += [head, "|" + "---|" * (len(MICS) + 4)]
    rank = sorted(have, key=lambda r: data[r[2]].get(1, float("inf")))
    best = {m: min(data[r][m] for _, _, r in have if m in data[r]) for m in MICS}
    for lbl, grp, run in rank:
        c = data[run]
        cells = " | ".join(
            (f"**{c[m]:.4f}**" if abs(c[m] - best[m]) < 1e-9 else f"{c[m]:.4f}") if m in c else "—"
            for m in MICS)
        ret = f"{c[1] / c[8]:.2f}x" if 1 in c and 8 in c else "—"
        L.append(f"| {lbl} | {ORIGIN[grp]} | {TRAIN_COL[grp]} | {cells} | {ret} |")
    if missing:
        L += ["", f"_pending: {', '.join(missing)}_"]
    open(f"{HERE}/table.md", "w").write("\n".join(L) + "\n")

    json.dump({run: data[run] for _, _, run in have}, open(f"{HERE}/curves.json", "w"), indent=2)

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False   # default font has no U+2212
    style = {"novd": ("-", 2.0), "prior": ("--", 1.4), "vdrop": ("-.", 1.8)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, logy in ((axes[0], False), (axes[1], True)):
        for lbl, grp, run in have:
            c = data[run]
            xs = [m for m in MICS if m in c]
            ax.plot(xs, [c[m] for m in xs], style[grp][0], lw=style[grp][1], marker="o", ms=4, label=lbl)
        ax.invert_xaxis()
        ax.set_xlabel("live mics"); ax.set_ylabel("test MAE (m)")
        ax.grid(alpha=.3); ax.spines[["top", "right"]].set_visible(False)
        if logy:
            # plain decimal ticks: the default LogFormatter renders 10^-1 with U+2212, which the
            # bundled font lacks, and the substituted glyph silently mangles the exponent.
            from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0), numticks=12))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_title("log scale")
        else:
            ax.set_title("linear scale")
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Mic failure at inference — every model trained on 8 mics (Replica test)\n"
                 "solid: ours, no mic-drop in training · dashed: prior work · "
                 "dash-dot: trained WITH mic drop")
    fig.tight_layout()
    fig.savefig(f"{HERE}/curve.png", dpi=110)
    print("\n".join(L))
    print(f"\n[saved] {HERE}/table.md, curves.json, curve.png")


if __name__ == "__main__":
    main()
