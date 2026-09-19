"""fig15: the four candidate explanations for the hop gain, and where the gain actually lands."""
import os, re, json
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
_RP = json.load(open("comparison_0820/compare.json"))
_MP = json.load(open("comparison_0820/mp3d_eval/compare.json"))

def te(n):
    for st in (_RP, _MP):
        if n in st: return st[n]["MAE"]
    return None

def bv(n):
    p = f"comparison_0820/logs/{n}.log"
    if not os.path.exists(p): return None
    v = [float(m.group(1)) for L in open(p) for m in [re.search(r"val_MAE=([\d.]+)m", L)] if m]
    return min(v) if v else None

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.4, 5.0),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})

# ---- left: the ruled-out explanations, Replica 4ch test
BASE = te("0820_sslamllrd_cs_fb_rep")
ROWS = [("hop 44  (설명 대상)",            te("0820_h44_sslcs_fb_rep"),   "#2e7d5b"),
        ("토큰 재배분 32f×16t",             te("0820_s9_tok832_h160_rep"), "#b9b2a4"),
        ("윈도 400→100 (분해능 4배)",        te("0820_s9_w100_h160_rep"),   "#b8544f")]
ROWS = [(l, v, c) for l, v, c in ROWS if v]
ys = range(len(ROWS))
ax.barh(list(ys), [v - BASE for _, v, _ in ROWS], color=[c for _, _, c in ROWS], height=.55)
ax.set_yticks(list(ys)); ax.set_yticklabels([l for l, _, _ in ROWS], fontsize=9)
ax.invert_yaxis(); ax.axvline(0, color="#22201c", lw=.9)
for i, (_, v, _) in enumerate(ROWS):
    d = v - BASE
    inside = d < -0.004
    ax.text(d + (0.002 if d > 0 else (0.002 if inside else -0.002)), i, f"{d:+.4f}",
            va="center", ha="left", fontsize=8.5, color="white" if inside else "#22201c")
ax.set_xlabel("Δ test MAE vs 기준선 0.2560      ← 개선")
ax.set_title("hop 44의 이득은 무엇으로 설명되는가 (Replica 4ch)\n"
             "토큰 배분도 시간 분해능도 아니다", fontsize=10.5)
ax.set_xlim(-0.012, 0.098); ax.grid(axis="x", alpha=.25)
ax.spines[["top", "right"]].set_visible(False)

# ---- right: path separation
ARMS = [("(d) fine 160 · AFM 160", "0820_sslamllrd_cs_fb_rep", "#6b675f"),
        ("(a) fine  44 · AFM  44", "0820_h44_sslcs_fb_rep",    "#4a7fb5"),
        ("(c) fine  44 · AFM 거침", "0820_s10_afm18_rep",       "#b8544f"),
        ("(b) fine 거침 · AFM  44", "0820_s10_fine18_rep",      "#2e7d5b")]
vals = [(l, te(n) or bv(n), te(n) is not None) for l, n, _ in ARMS]
cols = [c for _, _, c in ARMS]
ys = range(len(vals))
bx.barh(list(ys), [v for _, v, _ in vals], color=cols, height=.58)
bx.set_yticks(list(ys)); bx.set_yticklabels([l for l, _, _ in vals], fontsize=9)
bx.invert_yaxis()
for i, (_, v, is_test) in enumerate(vals):
    bx.text(v + .0012, i, f"{v:.4f}" + ("" if is_test else "  (val)"), va="center", fontsize=8.5)
bx.axvline(vals[0][1], color="#22201c", lw=.8, ls=":")
bx.set_xlim(0.240, 0.322); bx.set_xlabel("Replica 4ch MAE (m)")
bx.set_title("이득은 어느 가지의 것인가\nAFM에서 뺏으면 손해 · fine CNN에서 뺏으면 이득", fontsize=10.5)
bx.grid(axis="x", alpha=.25); bx.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("0820_report/figs/fig15_input_anatomy.png", dpi=120)
print("[saved] fig15", len(ROWS), "explanations,", len(vals), "arms")
