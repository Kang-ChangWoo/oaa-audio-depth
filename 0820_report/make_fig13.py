"""fig13: paired ConvStem ablation (LLRD fixed, stem linear -> conv) + MP3D 8ch fusion-axis curves."""
import os, re, json, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]

rep = json.load(open("comparison_0820/compare.json"))
mp = json.load(open("comparison_0820/mp3d_eval/compare.json"))
PAIRS = [  # (label, dict, no-cs run, +cs run)
    ("MP3D 8ch · eat", mp, "0820_eatllrd_r8_mp3d", "0820_eatllrd_cs_r8_mp3d"),
    ("MP3D 6ch · sslam", mp, "0820_sslam_llrd_r6_mp3d", "0820_sslamllrd_cs_r6_mp3d"),
    ("MP3D 4ch · sslam", mp, "0820_sslam_llrd_fb_mp3d", "0820_sslamllrd_cs_fb_mp3d"),
    ("MP3D 8ch · sslam", mp, "0820_sslam_llrd_r8_mp3d", "0820_sslamllrd_cs_r8_mp3d"),
    ("Rep 2ch · sslam", rep, "0820_sslam_llrd_r2_rep", "0820_sslamllrd_cs_r2_rep"),
    ("MP3D 2ch · eat", mp, "0820_eatllrd_r2_mp3d", "0820_eatllrd_cs_r2_mp3d"),
    ("Rep 6ch · eat", rep, "0820_eatllrd_r6_rep", "0820_eatllrd_cs_r6_rep"),
    ("MP3D 2ch · sslam", mp, "0820_sslam_llrd_r2_mp3d", "0820_sslamllrd_cs_r2_mp3d"),
    ("Rep 8ch · eat", rep, "0820_eatllrd_r8_rep", "0820_eatllrd_cs_r8_rep"),
    ("Rep 4ch · sslam", rep, "0820_sslam_llrd_fb_rep", "0820_sslamllrd_cs_fb_rep"),
    ("Rep 6ch · sslam", rep, "0820_sslam_llrd_r6_rep", "0820_sslamllrd_cs_r6_rep"),
    ("Rep 2ch · eat", rep, "0820_eatllrd_r2_rep", "0820_eatllrd_cs_r2_rep"),
    ("Rep 4ch · eat", rep, "0820_eatllrd_fb_rep", "0820_eatllrd_cs_fb_rep"),
    ("MP3D 6ch · eat", mp, "0820_eatllrd_r6_mp3d", "0820_eatllrd_cs_r6_mp3d"),
]
rows = [(l, d[b]["MAE"] - d[a]["MAE"]) for l, d, a, b in PAIRS if a in d and b in d]

def curve(n):
    out = {}
    p = f"comparison_0820/logs/{n}.log"
    for suf in ("", ".cancelled"):
        if os.path.exists(p + suf):
            for L in open(p + suf):
                m = re.search(r"\[ep (\d+)\].*val_MAE=([\d.]+)m", L)
                if m: out[int(m.group(1))] = float(m.group(2))
            break
    return out

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.4, 5.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})

ys = range(len(rows))
cols = ["#2e7d5b" if v < -0.01 else ("#b8544f" if v > 0.01 else "#b9b2a4") for _, v in rows]
ax.barh(list(ys), [v for _, v in rows], color=cols, height=.62)
ax.set_yticks(list(ys)); ax.set_yticklabels([l for l, _ in rows], fontsize=8.5)
ax.invert_yaxis(); ax.axvline(0, color="#22201c", lw=.9)
for x in (-0.01, 0.01):
    ax.axvline(x, color="#22201c", lw=.7, ls=":")
for i, (_, v) in enumerate(rows):
    ax.text(v + (0.004 if v > 0 else -0.004), i, f"{v:+.3f}", va="center",
            ha="left" if v > 0 else "right", fontsize=7.5)
ax.set_xlabel("ΔMAE (m):  ConvStem − linear patch embed        ← ConvStem 우세")
ax.set_title("ConvStem 페어드 제거실험 — 8승 5무 1패", fontsize=10.5)
ax.set_xlim(-0.29, 0.10); ax.grid(axis="x", alpha=.25)
ax.spines[["top", "right"]].set_visible(False)

SERIES = [("A0  stem=conv (기준)", "0820_sslamllrd_cs_r8_mp3d", "#6b675f", "-"),
          ("A1  conv_res (취소)", "0820_sslcs_A1_r8_mp3d", "#c9a227", "--"),
          ("A2  conv_ms", "0820_sslcs_A2_r8_mp3d", "#b8544f", "--"),
          ("B2  gated mic-differential", "0820_sslcs_B2_r8_mp3d", "#2e7d5b", "-")]
for lbl, run, col, ls in SERIES:
    c = curve(run)
    if not c: continue
    xs = sorted(c)
    bx.plot(xs, [c[e] for e in xs], ls, color=col, lw=2.1, marker="o", ms=3.4,
            label=f"{lbl} — best {min(c.values()):.4f}")
bx.axhline(0.7467, color="#22201c", lw=.9, ls=":")
bx.text(0.4, 0.7467, " OAA-CNN test 0.7467", fontsize=7.6, va="bottom", color="#22201c")
bx.set_xlabel("epoch"); bx.set_ylabel("val MAE (m)")
bx.set_title("MP3D 8ch — stem 축은 막히고 fusion 축만 열린다", fontsize=10.5)
bx.legend(fontsize=8, frameon=False); bx.grid(alpha=.25)
bx.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("0820_report/figs/fig13_convstem_fusion.png", dpi=120)
print("[saved] 0820_report/figs/fig13_convstem_fusion.png", len(rows), "pairs")
