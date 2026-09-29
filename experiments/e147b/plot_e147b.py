#!/usr/bin/env python3
"""E147b figure: FRAC 0.25 vs 0.50 (vs 1.00 trend), <0.5 m and >4 m F1 vs number of positions."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "AppleGothic"
rcParams["axes.unicode_minus"] = False        # ASCII hyphen, not U+2212

D = json.load(open("/tmp/e147b/plotdata_e147b.json"))
C = D["curves"]
NS = {"L24": [1, 2, 4, 8, 16, 24], "L32": [1, 2, 4, 8, 16, 32]}


def curve(lad, frac, bin_):
    return [C[f"{lad}|M1|{frac}|{bin_}|{n}"] for n in NS[lad]]


FR = [("0.25", "#1f4e79", "o", "-"), ("0.50", "#c0392b", "s", "-"), ("1.00", "#7f8c8d", "^", "--")]
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), sharex=False)

for col, lad in enumerate(("L32", "L24")):
    top = NS[lad][-1]
    for row, (bin_, title) in enumerate(((("<0.5"), "근거리 (0.5 m 미만)"), ((">4"), "원거리 (4 m 초과)"))):
        ax = axes[row][col]
        for frac, c, mk, ls in FR:
            y = curve(lad, frac, bin_)
            ax.plot(NS[lad], y, ls, color=c, marker=mk, ms=6, lw=2.0,
                    label=f"FRAC {frac}", zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(NS[lad])
        ax.set_xticklabels([str(n) for n in NS[lad]])
        ax.grid(alpha=0.25, zorder=0)
        n_seq = 8 if lad == "L32" else 15
        ax.set_title(f"{title}   |   {lad} (T>={top}, {n_seq}개 고정 코호트)", fontsize=11)
        ax.set_ylabel("F1")
        if row == 1:
            ax.set_xlabel("위치 관측 수 N")
        if bin_ == "<0.5":
            # annotate the peak-to-top drop for 0.25 and 0.50
            for frac, c, _, _ in FR[:2]:
                y = curve(lad, frac, bin_)
                i8 = NS[lad].index(8)
                d = y[i8] - y[-1]
                ax.annotate("", xy=(top, y[-1]), xytext=(top, y[i8]),
                            arrowprops=dict(arrowstyle="->", color=c, lw=1.6))
                ax.text(top * 1.08, (y[i8] + y[-1]) / 2, f"D={d:+.4f}".replace("−", "-"),
                        color=c, fontsize=9, va="center")
            ax.set_xlim(right=top * 2.6)
        ax.legend(fontsize=9, loc="best")

fig.suptitle("E147b — 근/원거리 trade는 FRAC 예산 artifact가 아니다: 예산을 늘리면 손실이 오히려 커진다\n"
             "(고정 기준 재채점, M1 채점, r2, 시드 0/1/2, GPU 미사용)", fontsize=12.5)
fig.tight_layout(rect=[0, 0.0, 1, 0.93])
out = "/tmp/e147b/E147b_frac_sensitivity.png"
fig.savefig(out, dpi=170)
print("wrote", out)
