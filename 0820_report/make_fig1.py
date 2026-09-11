#!/usr/bin/env python3
"""fig1 channel scaling — regenerate from the current confirmed matrix (multi-seed cells = seed means).
Values mirror the family-best rows of the report table / 0820_RESULTS.md."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CH = [2, 4, 6, 8]
SERIES = {  # (replica, mp3d), None = still training
    "OAA-CNN (paper)":  ([0.2894, 0.2596, 0.2384, 0.2368], [0.9084, 0.7849, 0.7502, 0.7467], "#404040"),
    "eat+LLRD":         ([0.2762, 0.2609, 0.2427, 0.2371], [0.9018, 0.7717, 0.7736, 0.7395], "#e07b28"),
    "sslam(+LLRD)":     ([0.2711, 0.2659, 0.2336, 0.2363], [0.8965, 0.7803, 0.7848, 0.9782], "#1f7a54"),
    "EchoDiffusion":    ([0.2854, 0.2695, 0.2556, 0.2600], [0.9007, 0.7928, 0.7786, 0.7572], "#7d6fc2"),
}
fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.6))
for ax, idx, title in [(axes[0], 0, "Replica"), (axes[1], 1, "MP3D")]:
    for name, tup in SERIES.items():
        ys = tup[idx]; c = tup[2]
        xs = [c_ for c_, y in zip(CH, ys) if y is not None]
        vs = [y for y in ys if y is not None]
        ax.plot(xs, vs, "-o", color=c, label=name, lw=2, ms=5)
    ax.set_title(title); ax.set_xticks(CH); ax.set_xticklabels([f"{c}ch" for c in CH])
    ax.set_ylabel("test MAE (m)"); ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Channel scaling: 관측 수 × 데이터셋 (낮을수록 좋음; 다시드 칸=시드 평균)")
fig.tight_layout()
fig.savefig("figs/fig1_channel_scaling.png", dpi=110, bbox_inches="tight")
print("saved figs/fig1_channel_scaling.png")
