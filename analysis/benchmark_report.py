"""Build analysis/report_final_benchmark.md — the 8-cell table for the current best AFM candidate.

Every number is read from the campaign's own compare.json files; nothing is re-derived or typed in
except the OAA-CNN and EchoDiffusion baseline run names, which are resolved by exact MAE match
against the released comparison/ directories and printed so the mapping is auditable.

  python3 analysis/benchmark_report.py
"""
import os, sys, json, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

TIE = 0.01          # campaign rule: |dMAE| < 0.01 is a tie, not a win or a loss
M = ["MAE", "RMSE", "AbsRel", "delta1", "near<3", "mid3-6", "far>6"]

REP = json.load(open("comparison_0820/compare.json"))
MP  = json.load(open("comparison_0820/mp3d_eval/compare.json"))
OLDR = json.load(open("comparison/compare.json"))
OLDM = json.load(open("comparison_mp3d/compare.json"))
ECOR = json.load(open("comparison/compare_eco.json"))
ECOM = json.load(open("comparison_mp3d/compare_eco.json"))

CELLS = [("Replica", 2, "r2"), ("Replica", 4, "fb"), ("Replica", 6, "r6"), ("Replica", 8, "r8"),
         ("MP3D", 2, "r2"), ("MP3D", 4, "fb"), ("MP3D", 6, "r6"), ("MP3D", 8, "r8")]

OURS = {("Replica", "r2"): "0820_h44_sslcs_r2_rep", ("Replica", "fb"): "0820_h44_sslcs_fb_rep",
        ("Replica", "r6"): "0820_h44_sslcs_r6_rep", ("Replica", "r8"): "0820_h44_sslcs_r8_rep",
        ("MP3D", "r2"): "0820_h44_sslcs_r2_mp3d", ("MP3D", "fb"): "0820_h44_sslcs_fb_mp3d",
        ("MP3D", "r6"): "0820_h44_sslcs_r6_mp3d", ("MP3D", "r8"): "0820_h44_sslcs_r8_mp3d"}
CNN = {("Replica", "r2"): ("oaa_r2_fin", OLDR), ("Replica", "fb"): ("oaa_fb_fin", OLDR),
       ("Replica", "r6"): ("oaa_r6_fin", OLDR), ("Replica", "r8"): ("oaa_r8_fin", OLDR),
       ("MP3D", "r6"): ("oaa_r6_fin", OLDM), ("MP3D", "r8"): ("oaa_r8_fin", OLDM)}
ECO = {("Replica", "r2"): ("eco_r2_fin", ECOR), ("Replica", "fb"): ("eco_fb_fin", ECOR),
       ("Replica", "r6"): ("eco_r6_fin", ECOR), ("Replica", "r8"): ("eco_r8_fin", ECOR),
       ("MP3D", "r2"): ("eco_r2_wstd", ECOM), ("MP3D", "fb"): ("eco_fb_wstd", ECOM),
       ("MP3D", "r6"): ("eco_r6", ECOM), ("MP3D", "r8"): ("eco_r8", ECOM)}
for cell, tgt in ((("MP3D", "r2"), 0.9084), (("MP3D", "fb"), 0.7849)):      # resolve by exact MAE
    hits = [k for k in OLDM if abs(OLDM[k].get("MAE", 9) - tgt) < 5e-5]
    CNN[cell] = (hits[0] if hits else None, OLDM)


def get(tbl, cell):
    k, st = tbl.get(cell, (None, None))
    return st.get(k) if (k and st and k in st) else None


def ours(cell):
    st = REP if cell[0] == "Replica" else MP
    return st.get(OURS[cell])


def main():
    lines, missing = [], []
    print(f"{'cell':12s}{'OAA-CNN run':22s}{'Eco run':16s}{'ours run':28s}")
    for ds, ch, mode in CELLS:
        c = (ds, mode)
        print(f"{ds+' '+str(ch)+'ch':12s}{str(CNN.get(c,(None,))[0]):22s}{str(ECO.get(c,(None,))[0]):16s}{OURS[c]:28s}")
        if ours(c) is None: missing.append(f"{ds} {ch}ch")
    lines += ["# Final benchmark — current best AFM candidate across all eight cells", "",
              "Model: **sslam + LLRD + conv stem, STFT hop 44**, one fixed configuration, single seed",
              "unless stated. Numbers are read from the campaign's compare.json files; the baseline run",
              "names resolved for each cell are listed at the end so the mapping is auditable.", "",
              f"Tie rule: |dMAE| < {TIE} is a tie (the campaign's existing threshold).", "",
              "## Results (test MAE, metres)", "",
              "| Dataset | Ch | EchoDiffusion | OAA-CNN | SSLAM+LLRD+CS+h44 | vs CNN | vs Eco |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    agg = {"all": [], "Replica": [], "MP3D": []}
    wl = {"cnn": [0, 0, 0], "eco": [0, 0, 0]}
    for ds, ch, mode in CELLS:
        c = (ds, mode)
        o, cn, ec = ours(c), get(CNN, c), get(ECO, c)
        f = lambda d: "pending" if d is None else f"{d['MAE']:.4f}"
        dc = de = ""
        if o and cn:
            g = o["MAE"] - cn["MAE"]; dc = f"{g:+.4f}"
            wl["cnn"][0 if g < -TIE else (2 if g > TIE else 1)] += 1
        if o and ec:
            g = o["MAE"] - ec["MAE"]; de = f"{g:+.4f}"
            wl["eco"][0 if g < -TIE else (2 if g > TIE else 1)] += 1
        if o:
            agg["all"].append(o["MAE"]); agg[ds].append(o["MAE"])
        lines.append(f"| {ds} | {ch} | {f(ec)} | {f(cn)} | **{f(o)}** | {dc or '—'} | {de or '—'} |")
    lines += ["", "## Aggregates", ""]
    for k, v in agg.items():
        lines.append(f"- {k} mean MAE: {'pending' if len(v) < (8 if k=='all' else 4) else f'{sum(v)/len(v):.4f}'} ({len(v)} cells)")
    lines += ["", f"- vs OAA-CNN: **{wl['cnn'][0]} win / {wl['cnn'][1]} tie / {wl['cnn'][2]} loss**",
              f"- vs EchoDiffusion: **{wl['eco'][0]} win / {wl['eco'][1]} tie / {wl['eco'][2]} loss**"]
    if missing:
        lines += ["", f"_pending cells: {', '.join(missing)}_"]
    # ---- matched-hop control: both baselines retrained on the same input recipe
    H44C = {("Replica","r2"):"0820_h44_cnn_r2_rep", ("Replica","fb"):"0820_h44_cnn_fb_rep",
            ("Replica","r6"):"0820_h44_cnn_r6_rep", ("Replica","r8"):"0820_h44_cnn_r8_rep",
            ("MP3D","r2"):"0820_h44_cnn_r2_mp3d", ("MP3D","fb"):"0820_h44_cnn_fb_mp3d",
            ("MP3D","r6"):"0820_h44_cnn_r6_mp3d", ("MP3D","r8"):"0820_h44_cnn_r8_mp3d_s1"}
    H44E = {c: f"0820_h44_eco_{m}_{'rep' if d=='Replica' else 'mp3d'}" for (d, m) in
            [("Replica","r2"),("Replica","fb"),("Replica","r6"),("Replica","r8"),
             ("MP3D","r2"),("MP3D","fb"),("MP3D","r6"),("MP3D","r8")] for c in [(d, m)]}
    lines += ["", "## Matched-hop control (every model at hop 44)", "",
              "The table above reads our hop-44 model against baselines trained at hop 160, which prices",
              "the input recipe together with the encoder. This block retrains both baselines on the same",
              "input; cells still training read pending.", "",
              "| Dataset | Ch | OAA-CNN @44 | EchoDiffusion @44 | ours @44 | vs CNN@44 | vs Eco@44 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    mw = {"cnn": [0,0,0], "eco": [0,0,0]}
    for ds, ch, mode in CELLS:
        c = (ds, mode); st = REP if ds == "Replica" else MP
        o = ours(c)
        cn = st.get(H44C[c]); ec = st.get(H44E[c])
        f = lambda d: "pending" if d is None else f"{d['MAE']:.4f}"
        dc = de = "—"
        if o and cn:
            g = o["MAE"] - cn["MAE"]; dc = f"{g:+.4f}"
            mw["cnn"][0 if g < -TIE else (2 if g > TIE else 1)] += 1
        if o and ec:
            g = o["MAE"] - ec["MAE"]; de = f"{g:+.4f}"
            mw["eco"][0 if g < -TIE else (2 if g > TIE else 1)] += 1
        lines.append(f"| {ds} | {ch} | {f(cn)} | {f(ec)} | **{f(o)}** | {dc} | {de} |")
    lines += ["", f"- matched-hop vs OAA-CNN: **{mw['cnn'][0]} win / {mw['cnn'][1]} tie / {mw['cnn'][2]} loss** "
                  f"({sum(mw['cnn'])} of 8 cells measured)",
              f"- matched-hop vs EchoDiffusion: **{mw['eco'][0]} win / {mw['eco'][1]} tie / {mw['eco'][2]} loss** "
              f"({sum(mw['eco'])} of 8 cells measured)"]

    lines += ["", "## Baseline run names resolved per cell", "", "| cell | OAA-CNN | EchoDiffusion | ours |", "|---|---|---|---|"]
    for ds, ch, mode in CELLS:
        c = (ds, mode)
        lines.append(f"| {ds} {ch}ch | `{CNN.get(c,(None,))[0]}` | `{ECO.get(c,(None,))[0]}` | `{OURS[c]}` |")
    open("analysis/report_final_benchmark.md", "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[8:30]))
    print(f"\n[saved] analysis/report_final_benchmark.md")


if __name__ == "__main__":
    main()
