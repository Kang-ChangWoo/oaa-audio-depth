"""comparison_mp3d/compare.json (+eco rows) -> results_mp3d.tex (all-metric table + experiment settings).

eco rows are merged from values measured with eval_echodiffusion.py (compare.json only holds eval.py-family runs).
Run:  python tools/make_mp3d_tex.py
"""
# --- repo-root bootstrap: importable root modules (eval, data_*, model) + relative comparison/ paths
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
_os.chdir(ROOT)
import json, os, datetime

HERE = ROOT
CJ = json.load(open(f"{HERE}/comparison_mp3d/compare.json"))

# eco rows measured with eval_echodiffusion.py (2823-sample window evaluation, measured 2026-07-24/25)
ECO = {
    "eco_r2_wstd":  dict(MAE=0.9007, MAE_plain=0.7537, RMSE=1.3805, AbsRel=0.5259, log10=0.1758, delta1=0.4528, delta2=0.6730, delta3=0.8012, P=156.1),
    "eco_r2_wnone": dict(MAE=0.9251, MAE_plain=0.7736, RMSE=1.3815, AbsRel=0.5807, log10=0.1802, delta1=0.4429, delta2=0.6622, delta3=0.7937, P=60.1),
    "eco_fb_wall":  dict(MAE=0.7789, MAE_plain=0.6609, RMSE=1.2611, AbsRel=0.3993, log10=0.1460, delta1=0.5328, delta2=0.7463, delta3=0.8533, P=156.1),
    "eco_fb_wnone": dict(MAE=0.7820, MAE_plain=0.6639, RMSE=1.2717, AbsRel=0.3826, log10=0.1460, delta1=0.5294, delta2=0.7448, delta3=0.8531, P=60.1),
    "eco_fb_wstd":  dict(MAE=0.7928, MAE_plain=0.6700, RMSE=1.2889, AbsRel=0.3993, log10=0.1495, delta1=0.5257, delta2=0.7384, delta3=0.8468, P=156.1),
    "eco_r6_wnone": dict(MAE=0.7550, MAE_plain=0.6465, RMSE=1.2477, AbsRel=0.3777, log10=0.1403, delta1=0.5488, delta2=0.7604, delta3=0.8614, P=60.1),
    "eco_r6_wall":  dict(MAE=0.7681, MAE_plain=0.6520, RMSE=1.2548, AbsRel=0.3934, log10=0.1435, delta1=0.5457, delta2=0.7540, delta3=0.8571, P=156.1),
    "eco_r8_wall":  dict(MAE=0.7357, MAE_plain=0.6275, RMSE=1.2119, AbsRel=0.3790, log10=0.1365, delta1=0.5603, delta2=0.7653, delta3=0.8676, P=156.1),
}

# (display name, source key, is_eco, channels, footnote) — ◇ = trained with the 2799 window (evaluation is 2823 for all rows)
ROWS = [
    ("2ch", [
        ("EchoDiffusion",               "eco_r2_wstd",  True,  "◇"),   # 2ch: wstd=wall (only 2 channels) = representative row
        ("OAA plain s0",                "REL_2ch_adaln_s0", False, ""),
        ("OAA plain s1",                "REL_2ch_adaln_s1", False, ""),
        ("ViT",                         "vit_r2",       False, "◇"),
        ("BatVision",                   "bat_r2_fin",   False, ""),
        ("EchoScan",                    "es_r2_fin",    False, ""),
        ("EchoDiffusion (no-wave abl.)","eco_r2_wnone", True,  "◇"),
        ("OAA fullres-pkg",             "oaa_r2",       False, "◇"),
        ("ResNet",                      "rn_r2",        False, "◇"),
    ]),
    ("4ch", [
        ("OAA fullres s0 (cB)",         "OAA_fullres_lr5e-4_s0", False, ""),
        ("OAA fullres s1 (cB)",         "OAA_fullres_lr5e-4_s1", False, ""),
        ("EchoDiffusion (fb)",          "eco_fb_wall",  True,  "◇"),   # representative row = wall
        ("EchoDiffusion (no-wave abl.)","eco_fb_wnone", True,  "◇"),
        ("OAA champion (cB)",           "REL_B_R3_s0",  False, ""),
        ("OAA champion s1 (cB)",        "REL_C_U0_s0",  False, ""),
        ("EchoDiffusion (wstd abl.)",   "eco_fb_wstd",  True,  "◇"),
        ("OAA plain (fs=0+90)",         "OAA_sub_AB_s0", False, ""),
        ("OAA plain (fb=0+180)",        "OAA_sub_cA_s0", False, ""),
        ("BatVision (fb)",              "bat_fb_fin",   False, ""),
        ("ResNet (fb)",                 "rn_fb",        False, ""),
        ("ViT (fb)",                    "vit_fb_fin",   False, ""),
        ("EchoScan (fb)",               "es_fb_fin",    False, ""),
    ]),
    ("6ch", [
        ("OAA s1",                      "OAA_r6_adaln_s1", False, ""),
        ("OAA s0",                      "OAA_r6_adaln_s0", False, ""),
        ("EchoDiffusion (no-wave abl.)","eco_r6_wnone", True,  ""),
        ("EchoDiffusion",               "eco_r6_wall",  True,  ""),   # representative row = wall (2026-07-25)
        ("BatVision",                   "bat_r6_fin",   False, ""),
        ("ViT",                         "vit_r6_fin",   False, ""),
        ("ResNet",                      "rn_r6_fin",    False, ""),
        ("EchoScan",                    "es_r6_fin",    False, ""),
    ]),
    ("8ch", [
        ("OAA (+silog)",                "OAA_r8_adaln_silog_s0", False, ""),
        ("OAA s1",                      "OAA_r8_adaln_s1", False, ""),
        ("EchoDiffusion",               "eco_r8_wall",  True,  ""),   # representative row = wall
        ("BatVision",                   "bat_r8_fin",   False, ""),
        ("ViT",                         "vit_r8_fin",   False, ""),
        ("ResNet",                      "rn_r8_fin",    False, ""),
        ("EchoScan",                    "es_r8_fin",    False, ""),
    ]),
]

def get(key, eco):
    if eco:
        return ECO.get(key)
    v = CJ.get(key)
    if v is None:
        return None
    return dict(MAE=v["MAE"], MAE_plain=v["MAE_plain"], RMSE=v["RMSE"], AbsRel=v["AbsRel"],
                log10=v["log10"], delta1=v["delta1"], delta2=v["delta2"], delta3=v["delta3"],
                P=v["Params(M)"])

L = []
L.append("% Auto-generated by tools/make_mp3d_tex.py — " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
L.append(r"\begin{table*}[t]\centering")
L.append(r"\caption{Matterport3D test-set comparison (3{,}600 scene-disjoint samples). All rows evaluated under the identical pipeline (magnitude STFT, nearest resize, 2{,}823-sample round-trip window, cos-latitude-weighted per-image metrics). $\diamond$: trained with the 2{,}799-sample window (evaluation identical; measured sensitivity $\le$0.005).}")
L.append(r"\label{tab:mp3d}")
L.append(r"\small\begin{tabular}{llrrrrrrrrr}")
L.append(r"\toprule")
L.append(r"Ch & Method & MAE$\downarrow$ & MAE$_{\mathrm{plain}}\downarrow$ & RMSE$\downarrow$ & AbsRel$\downarrow$ & $\log_{10}\downarrow$ & $\delta_1\uparrow$ & $\delta_2\uparrow$ & $\delta_3\uparrow$ & Params(M) \\")
for ch, rows in ROWS:
    L.append(r"\midrule")
    vals = [(nm, get(k, e), fn) for nm, k, e, fn in rows]
    have = [(nm, v, fn) for nm, v, fn in vals if v]
    if not have:
        continue
    best = min(v["MAE"] for _, v, _ in have)
    for i, (nm, v, fn) in enumerate(sorted(have, key=lambda x: x[1]["MAE"])):
        mae = f"\\textbf{{{v['MAE']:.4f}}}" if abs(v["MAE"] - best) < 1e-9 else f"{v['MAE']:.4f}"
        chc = ch if i == 0 else ""
        L.append(f"{chc} & {nm}{fn} & {mae} & {v['MAE_plain']:.4f} & {v['RMSE']:.4f} & {v['AbsRel']:.4f} & "
                 f"{v['log10']:.4f} & {v['delta1']:.4f} & {v['delta2']:.4f} & {v['delta3']:.4f} & {v['P']:.1f} \\\\")
L.append(r"\bottomrule\end{tabular}\end{table*}")
L.append("")
L.append(r"% ---------------- Experimental settings ----------------")
L.append(r"\paragraph{Setup.} Matterport3D (0303renew render), 90 scenes, original scene-disjoint splits")
L.append(r"(28{,}800/3{,}540/3{,}600; samples with incomplete 4-yaw groups dropped: 3 val samples of ZMojNkEp431).")
L.append(r"Input: binaural recordings at 4 yaws (0/90/180/270$^\circ$); wav cut to 2{,}823 samples (10\,m round trip @340\,m/s,")
L.append(r"48\,kHz), magnitude STFT ($n_\mathrm{fft}{=}512$, win 400, hop 160), nearest-resized to $256{\times}512$.")
L.append(r"Target: radial ERP depth ($256{\times}512$, $\div$10\,m). Channel modes: r2=[0L,0R], fb=+180$^\circ$ pair,")
L.append(r"fs=+90$^\circ$ pair, cB=[0L,0R,90R,270L], r6=0/90/270 pairs, r8=all four pairs.")
L.append(r"\paragraph{Recipes.} All models: masked L1, AdamW, warmup+cosine, model selection on val.")
L.append(r"OAA plain: 30 ep, bs 32, lr $10^{-3}$, EMA 0.999. OAA fullres: 30 ep, bs 16, lr $5{\times}10^{-4}$, EMA.")
L.append(r"EchoDiffusion: 40 ep, bs 16, lr $10^{-4}$, bf16 (execution-fix port; native $256{\times}512$ decode granted;")
L.append(r"wall/wstd/no-wave = wave-branch channel ablation). BatVision: 40 ep, bs 64, lr $2{\times}10^{-3}$.")
L.append(r"ViT/ResNet: 40 ep, bs 24--48, lr $10^{-3}$, ImageNet init.")
open(f"{HERE}/comparison_mp3d/results_mp3d.tex", "w").write("\n".join(L))
print(f"written comparison_mp3d/results_mp3d.tex ({sum(1 for ch,rows in ROWS for r in rows)} row templates)")
