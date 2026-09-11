"""Merge the micdrop curves into the study table + figure.

  python3 analysis/micdrop_study/report.py

Reads analysis/micdrop_study/shards/*.json (written by run.sh) plus the released curves in
comparison/eval_micdrop.json, and writes curves.json, table.md and curve.png next to them.
Missing models are listed as pending and skipped, so this is safe to run mid-sweep.
"""
import os, sys, json, glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from analysis.micdrop_study.roster import ROSTER, ECO, RELEASED, RELEASED_ENTRIES

HERE = "analysis/micdrop_study"
MICS = list(range(8, 0, -1))          # 8..1 live mics  (k = 8 - mics)


def load():
    """-> {run: {mics: MAE}} from this study's shards, falling back to the released curves."""
    out = {}
    for f in glob.glob(f"{HERE}/shards/*.json") + ([RELEASED] if os.path.exists(RELEASED) else []):
        try:
            blob = json.load(open(f))
        except Exception:
            continue
        for run, rec in blob.items():
            if run in out or "curve" not in rec:      # a re-run shard wins over the released copy
                continue
            c = rec["curve"]
            out[run] = {8: c["k0"]} | {
                8 - k: c[f"k{k}"]["mean"] for k in range(1, 8) if f"k{k}" in c}
    return out


def main():
    data = load()
    seen, roster = set(), []
    for e in ROSTER + [ECO] + RELEASED_ENTRIES:      # de-dup: first spelling of a run wins
        if e[3] not in seen:
            seen.add(e[3]); roster.append(e)
    have = [e for e in roster if e[3] in data]
    missing = [e[3] for e in roster if e[3] not in data]

    # ---- one ranking over every model; mic-drop training is a column, not a separate section
    L = ["# Mic-drop degradation: every model trained on 8 mics, tested on 8-1", "",
         "Replica test (off3), MAE in metres. At inference k of the 8 channels are zeroed and the",
         "pose conditioning stays truthful -- the *mic failure* curve, not a smaller-rig retrain.",
         "Each k is the mean over 3 fixed-seed random subsets. `mic-drop` = did the model see",
         "dropped mics during TRAINING (among prior work only EchoDiffusion has such a variant).",
         "`ret@1` = MAE(1 mic)/MAE(8 mic): how much of the 8-mic error survives total rig loss --",
         "lower is a flatter, more graceful curve. Sorted by 1-mic MAE.", ""]
    L += ["| model | origin | mic-drop | " + " | ".join(f"{m}mic" for m in MICS) + " | ret@1 |",
          "|" + "---|" * (len(MICS) + 4)]
    rank = sorted(have, key=lambda e: data[e[3]].get(1, float("inf")))
    best = {m: min(data[e[3]][m] for e in have if m in data[e[3]]) for m in MICS}
    for lbl, origin, drop, run in rank:
        c = data[run]
        cells = " | ".join(
            (f"**{c[m]:.4f}**" if abs(c[m] - best[m]) < 1e-9 else f"{c[m]:.4f}") if m in c else "--"
            for m in MICS)
        ret = f"{c[1] / c[8]:.2f}x" if 1 in c and 8 in c else "--"
        L.append(f"| {lbl} | {'ours' if origin == 'ours' else 'prior work'} | "
                 f"{'yes' if drop else 'no'} | {cells} | {ret} |")
    if missing:
        L += ["", f"_pending: {', '.join(missing)}_"]
    open(f"{HERE}/table.md", "w").write("\n".join(L) + "\n")

    json.dump({e[3]: data[e[3]] for e in have}, open(f"{HERE}/curves.json", "w"), indent=2)

    # ---- figure: colour = origin, line style = mic-drop training
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    plt.rcParams["axes.unicode_minus"] = False       # bundled font has no U+2212
    cmap = {"ours": plt.get_cmap("winter"), "prior": plt.get_cmap("autumn")}
    n = {k: sum(1 for e in have if e[1] == k) or 1 for k in ("ours", "prior")}
    idx = {"ours": 0, "prior": 0}
    styled = []
    for lbl, origin, drop, run in sorted(have, key=lambda e: (e[1] != "ours", not e[2], e[0])):
        col = cmap[origin](idx[origin] / max(n[origin] - 1, 1) * 0.85)
        idx[origin] += 1
        styled.append((lbl, run, col, "-" if drop else "--"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, logy in ((axes[0], False), (axes[1], True)):
        for lbl, run, col, ls in styled:
            c = data[run]
            xs = [m for m in MICS if m in c]
            ax.plot(xs, [c[m] for m in xs], ls, color=col, lw=1.9, marker="o", ms=4, label=lbl)
        ax.invert_xaxis()
        ax.set_xlabel("live mics"); ax.set_ylabel("test MAE (m)")
        ax.grid(alpha=.3); ax.spines[["top", "right"]].set_visible(False)
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 3., 5.), numticks=12))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_title("log scale")
        else:
            ax.set_title("linear scale")
    axes[0].legend(fontsize=7.5, frameon=False, ncol=2)
    fig.suptitle("Mic failure at inference — every model trained on 8 mics (Replica test)\n"
                 "blue-green: ours · orange-red: prior work · solid: trained WITH mic drop · dashed: without")
    fig.tight_layout()
    fig.savefig(f"{HERE}/curve.png", dpi=110)
    print("\n".join(L))
    print(f"\n[saved] {HERE}/table.md, curves.json, curve.png")


if __name__ == "__main__":
    main()
