#!/usr/bin/env python3
"""E147 figure: position-count ladder for >4 m F1, cohort-fixed, with the rotation axis marked."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

D = json.loads(Path("/Users/orange-brown/.openclaw/workspace/tmp/e147/dl/plotdata_e147.json").read_text())
d = D["data"]
n = D["cohort_n"]

C_L24, C_L32, C_ROT = "#1f6feb", "#c0392b", "#8a8f98"
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.7), gridspec_kw={"width_ratios": [1.32, 1]})

# ---------------- panel 1: the answer -------------------------------------------------
for key, lab, col, mk in (("r2_L24", f"L24 · 위치 (T≥24, n={n['L24']})", C_L24, "o"),
                          ("r2_L32", f"L32 · 위치 (T≥32, n={n['L32']})", C_L32, "s")):
    x = np.array(d[key]["N"], float)
    y = np.array(d[key][">4|f1"], float)
    e = np.array(d[key][">4|f1|sd"], float)
    ax.errorbar(x, y, yerr=e, color=col, marker=mk, ms=7, lw=2.4, capsize=3, label=lab, zorder=4)
    ax.annotate(f"{y[-1]:.4f}", (x[-1], y[-1]), textcoords="offset points", xytext=(9, -3),
                color=col, fontsize=10, fontweight="bold")

for key, lab, col, mk in (("r8_L24", "L24 · 회전 r8 (관측 4배)", C_L24, "o"),
                          ("r8_L32", "L32 · 회전 r8 (관측 4배)", C_L32, "s")):
    x = np.array(d[key]["N"], float)
    y = np.array(d[key][">4|f1"], float)
    ax.plot(x, y, color=col, marker=mk, ms=4.5, lw=1.3, ls=":", alpha=.55, label=lab, zorder=3)

# rotation axis saturation, from the HEAR360 mic curve (4-5 headings)
ax.axvspan(4, 5, color=C_ROT, alpha=.20, zorder=1)
ax.text(4.5, 0.1665, "회전 축은\n여기서 포화\n(헤딩 4~5)", ha="center", va="top",
        fontsize=9.5, color="#4a4f57", linespacing=1.35)

ax.set_xscale("log", base=2)
ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 24, 32]))
ax.set_xticklabels(["1", "2", "4", "8", "16", "24", "32"])
ax.set_xlim(0.85, 44)
ax.set_ylim(-0.006, 0.175)
ax.set_xlabel("관측 위치 수 N  (로그 축)", fontsize=11.5)
ax.set_ylabel("> 4 m F1  (고정기준, 코호트 고정)", fontsize=11.5)
ax.set_title("위치를 늘려도 원거리 F1은 안 꺾인다 — 궤적 천장(32)까지 계속 오름",
             fontsize=12.5, fontweight="bold", pad=11)
ax.grid(alpha=.25, ls="--", lw=.6)
h, lb = ax.get_legend_handles_labels()
order = [2, 3, 0, 1]
ax.legend([h[i] for i in order], [lb[i] for i in order],
          fontsize=9, loc="upper left", framealpha=.93)
ax.axhline(0.005, color="#999", lw=.8, ls="-.")
ax.text(41, 0.0075, "포화 판정 문턱\n+0.005/스텝", ha="right", va="bottom", fontsize=8.5, color="#777")

# last-step delta, spelled out
ax.annotate("마지막 2배 스텝\n+0.0461", xy=(23.5, 0.132), xytext=(8.0, 0.158),
            fontsize=9.5, color=C_L32, fontweight="bold", ha="center", linespacing=1.3,
            arrowprops=dict(arrowstyle="->", color=C_L32, lw=1.2))

# ---------------- panel 2: what it costs ---------------------------------------------
k = "r2_L32"
x = np.array(d[k]["N"], float)
series = (("overall|f1", "전체 F1", "#2d3748", "-", "o"),
          ("<0.5|f1", "< 0.5 m (근거리)", "#1f9d55", "-", "^"),
          (">4|recall", "> 4 m recall", "#c0392b", "-", "s"),
          (">4|precision", "> 4 m precision", "#e08e0b", "--", "d"))
for key, lab, col, ls, mk in series:
    y = np.array(d[k][key], float)
    ax2.plot(x, y, color=col, ls=ls, marker=mk, ms=6, lw=2.1, label=lab)

ax2.set_xscale("log", base=2)
ax2.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32]))
ax2.set_xticklabels(["1", "2", "4", "8", "16", "32"])
ax2.set_xlim(0.85, 40)
ax2.set_ylim(-0.02, 0.86)
ax2.set_xlabel("관측 위치 수 N  (L32 코호트, n=8)", fontsize=11.5)
ax2.set_ylabel("값", fontsize=11.5)
ax2.set_title("원거리 이득은 전부 recall — 근거리는 오히려 손해",
              fontsize=12.5, fontweight="bold", pad=11)
ax2.grid(alpha=.25, ls="--", lw=.6)
ax2.legend(fontsize=9, loc="upper left", framealpha=.95, ncol=2, columnspacing=1.0)
ax2.annotate("8→32에서 근거리 -0.063", xy=(30, d[k]["<0.5|f1"][-1] + .012),
             xytext=(13.5, 0.695), fontsize=9.2, color="#15804a", ha="center",
             arrowprops=dict(arrowstyle="->", color="#1f9d55", lw=1.2))
ax2.annotate("precision 평평 (.22→.25)", xy=(17, d[k][">4|precision"][-2] - .012),
             xytext=(4.6, 0.085), fontsize=9.2, color="#b8780a", ha="center",
             arrowprops=dict(arrowstyle="->", color="#e08e0b", lw=1.2))

fig.suptitle("E147 · 위치 예산의 포화점 — 결론: 측정 범위(24·32) 안에서 포화하지 않는다",
             fontsize=14.5, fontweight="bold", y=0.995)
fig.text(0.5, 0.012,
         "실측 데이터 · EchoRecon r2/r8 posterior 캐시 + N-무관 고정기준(E142-fix), GPU 0, 39개 테스트 시퀀스 중 코호트 고정 · "
         "오차막대 = 선택 시드 0/1/2 모표준편차 · npred=0은 F1 0으로 채움(M1)",
         ha="center", fontsize=8.3, color="#666")
fig.tight_layout(rect=[0, 0.035, 1, 0.955])
out = "/Users/orange-brown/.openclaw/workspace/tmp/e147/E147_position_saturation.png"
fig.savefig(out, dpi=175, facecolor="white")
print("wrote", out)
