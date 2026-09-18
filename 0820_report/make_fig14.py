"""fig14: the input fix versus the fusion architectures on the MP3D 8ch loss cell, and the
hop-44 gain across every cell it has been run on."""
import os, re, json
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]

_MP = json.load(open("comparison_0820/mp3d_eval/compare.json"))
_RP = json.load(open("comparison_0820/compare.json"))

def bv(n):
    """test MAE (val 기준 순위가 test에서 뒤집히는 것을 이 캠페인이 반복 관측했으므로 test로 그린다)"""
    for st in (_MP, _RP):
        if n in st: return st[n]["MAE"]
    return None

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.0),
                             gridspec_kw={"width_ratios": [1.05, 1]})

# ---- left: MP3D 8ch, every intervention tried on the loss cell
BARS = [("A0  conv stem (기준)",        "0820_sslamllrd_cs_r8_mp3d", "#6b675f"),
        ("eat+LLRD+cs",                "0820_eatllrd_cs_r8_mp3d",   "#4a7fb5"),
        ("A2  stem 변형",               "0820_sslcs_A2_r8_mp3d",     "#b9b2a4"),
        ("B1  α-residual",              "0820_sslcs_B1_r8_mp3d",     "#b9b2a4"),
        ("C1  fine residual",           "0820_sslcs_C1_r8_mp3d",     "#b9b2a4"),
        ("B3  gate + 전역 ctx",          "0820_sslcs_B3_r8_mp3d",     "#b9b2a4"),
        ("rounds 3  (fusion 용량)",      "0820_tune_rounds3_r8_mp3d", "#b9b2a4"),
        ("native  (입력 레이아웃)",       "0820_sslnat_r8_mp3d",       "#c9a227"),
        ("C3  Model 3",                 "0820_sslcs_C3_r8_mp3d",     "#4a7fb5"),
        ("B2  gated differential",      "0820_sslcs_B2_r8_mp3d",     "#4a7fb5"),
        ("hop 44  (입력 시간해상도)",     "0820_h44_sslcs_r8_mp3d",    "#2e7d5b")]
vals = [(l, bv(r), c) for l, r, c in BARS]
vals = sorted([(l, v, c) for l, v, c in vals if v], key=lambda t: -t[1])
ys = range(len(vals))
ax.barh(list(ys), [v for _, v, _ in vals], color=[c for _, _, c in vals], height=.66)
ax.set_yticks(list(ys)); ax.set_yticklabels([l for l, _, _ in vals], fontsize=9)
ax.invert_yaxis()
base = dict((l, v) for l, v, _ in vals).get("A0  conv stem (기준)")
for i, (_, v, _) in enumerate(vals):
    ax.text(v + .006, i, f"{v:.4f}" + (f"  ({v-base:+.3f})" if base and abs(v-base) > 1e-9 else ""),
            va="center", fontsize=8)
ax.axvline(base, color="#22201c", lw=.9, ls=":")
ax.axvline(0.7467, color="#2e7d5b", lw=1.1, ls="--")
ax.text(0.7467, len(vals)-0.3, " OAA-CNN 0.7467", fontsize=7.6, color="#2e7d5b", va="top")
ax.set_xlim(0.60, 1.12); ax.set_xlabel("MP3D 8ch  test MAE (m)")
ax.set_title("캠페인 유일의 패배 칸에 시도한 모든 개입 (test)\n회색=무효 · 파랑=fusion 구조 · 초록=입력 · 점선=CNN 기준", fontsize=10.5)
ax.grid(axis="x", alpha=.25); ax.spines[["top", "right"]].set_visible(False)

# ---- right: hop-44 gain per cell (val deltas, same recipe)
PAIRS = [("MP3D 8ch", "0820_h44_sslcs_r8_mp3d", "0820_sslamllrd_cs_r8_mp3d"),
         ("MP3D 6ch", "0820_h44_sslcs_r6_mp3d", "0820_sslamllrd_cs_r6_mp3d"),
         ("MP3D 4ch", "0820_h44_sslcs_fb_mp3d", "0820_sslamllrd_cs_fb_mp3d"),
         ("MP3D 2ch", "0820_h44_sslcs_r2_mp3d", "0820_sslamllrd_cs_r2_mp3d"),
         ("Rep 8ch",  "0820_h44_sslcs_r8_rep",  "0820_sslamllrd_cs_r8_rep"),
         ("Rep 6ch",  "0820_h44_sslcs_r6_rep",  "0820_sslamllrd_cs_r6_rep"),
         ("Rep 4ch",  "0820_h44_sslcs_fb_rep",  "0820_sslamllrd_cs_fb_rep")]
rows = [(l, bv(a) - bv(b)) for l, a, b in PAIRS if bv(a) and bv(b)]
ys = range(len(rows))
bx.barh(list(ys), [d for _, d in rows],
        color=["#2e7d5b" if d < 0 else "#b8544f" for _, d in rows], height=.62)
bx.set_yticks(list(ys)); bx.set_yticklabels([l for l, _ in rows], fontsize=9)
bx.invert_yaxis(); bx.axvline(0, color="#22201c", lw=.9)
for i, (_, d) in enumerate(rows):
    inside = d < -0.05
    bx.text(d + (0.004 if d > 0 else (0.006 if inside else -0.004)), i, f"{d:+.4f}",
            va="center", ha="left" if (d > 0 or inside) else "right", fontsize=8,
            color="white" if inside else "#22201c")
bx.set_xlim(-0.28, 0.02)
bx.set_xlabel("Δ test MAE:  hop 44 − hop 160      ← hop 44 우세")
bx.set_title("입력 시간해상도의 효과는 MP3D에 집중된다", fontsize=10.5)
bx.grid(axis="x", alpha=.25); bx.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("0820_report/figs/fig14_input_vs_fusion.png", dpi=120)
print("[saved] fig14", len(vals), "bars,", len(rows), "cells")
