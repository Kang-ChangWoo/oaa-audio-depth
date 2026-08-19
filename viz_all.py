"""Qualitative grid for the comparison table: per-channel PNGs of test-sample depth maps.

Columns (paper order): GT | ResNet | ViT | Beyond(blank) | EchoScan | BatVision | EchoDiffusion | OAA.
Rows: 2 samples from each held-out test scene. One PNG per channel mode -> comparison/viz_all/.

Stage 1 (this script, base env): predict with every *base-env* _fin checkpoint, save npy + GT.
Stage 2 (viz_eco.py, isolated env): EchoDiffusion predictions -> npy.
Stage 3 (this script --compose): assemble PNGs; missing npy (byd, eco-not-yet) renders blank.

  REPLICA_ROOT=... DATA_MODULE=data_0422 python3 viz_all.py [--compose]
"""
import os, sys, argparse
os.environ.setdefault("DATA_MODULE", "data_0422")
import numpy as np
import torch

import data_0422 as dm
dm.ROOT = os.environ.get("REPLICA_ROOT", dm.ROOT)
import eval as ev

MAIN = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(MAIN, "comparison", "viz_all")
os.makedirs(OUT, exist_ok=True)

SAMPLES = [("apartment_2", 0), ("apartment_2", 220),
           ("frl_apartment_5", 0), ("frl_apartment_5", 220),
           ("office_4", 0), ("office_4", 220)]
MODES = ["r2", "fb", "r6", "r8"]
# checkpoint per (model, mode); None -> blank column. OAA 6/8ch: bmax stand-in until fe finishes.
CKPTS = {}
for m in MODES:
    CKPTS[("rn", m)] = f"comparison/rn_{m}_fin"
    CKPTS[("vit", m)] = f"comparison/vit_{m}_fin"
    CKPTS[("byd", m)] = None
    CKPTS[("es", m)] = f"comparison/es_{m}_fin"
    CKPTS[("bat", m)] = f"comparison/bat_{m}_fin"
    CKPTS[("eco", m)] = "EXTERNAL"                      # produced by viz_eco.py in the isolated env
CKPTS[("oaa", "r2")] = "comparison/oaa_r2_fin"
CKPTS[("oaa", "fb")] = "comparison/oaa_fb_fin"
CKPTS[("oaa", "r6")] = "comparison/deprecated/oaa_r6_bmax"   # temporary (fe pending)
CKPTS[("oaa", "r8")] = "comparison/deprecated/oaa_r8_bmax"


def predict():
    device = torch.device("cuda")
    # GT once (mode-independent)
    for si, (sc, st) in enumerate(SAMPLES):
        d, mask = dm._load_depth(f"{dm.ROOT}/{sc}/erp_depth/erp_depth_{st:03d}.npy")
        np.save(f"{OUT}/gt_{si}.npy", (d[0] * dm.MAX_DEPTH).numpy())
        np.save(f"{OUT}/mask_{si}.npy", mask[0].numpy())
    for (model, mode), path in CKPTS.items():
        if path in (None, "EXTERNAL"):
            continue
        f = os.path.join(MAIN, path, "best.pth")
        if not os.path.exists(f):
            print(f"[skip] {model}/{mode}: no ckpt at {path}", flush=True)
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        net, dmode, nch, kind, poses = ev.build(ck["args"])
        net.load_state_dict(ck["state_dict"]); net.to(device).eval()
        md = ck["args"].get("max_depth", 10.0)
        with torch.no_grad():
            for si, (sc, st) in enumerate(SAMPLES):
                wave = dm._load_wave(sc, st, mode)
                x = wave if kind == "wave" else dm._stft_mag(wave)
                x = x[:nch].unsqueeze(0).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    D = (net(x, view_poses=poses) if poses is not None else net(x)).float() * md
                np.save(f"{OUT}/pred_{model}_{mode}_{si}.npy", D[0, 0].cpu().numpy())
        del net; torch.cuda.empty_cache()
        print(f"[ok] {model}/{mode}", flush=True)


def compose():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("GT", "gt"), ("ResNet-50", "rn"), ("ViT-B/16", "vit"), ("Beyond", "byd"),
            ("EchoScan", "es"), ("BatVision", "bat"), ("EchoDiffusion", "eco"), ("OAA (ours)", "oaa")]
    for mode, nch in zip(MODES, [2, 4, 6, 8]):
        fig, axes = plt.subplots(len(SAMPLES), len(cols),
                                 figsize=(len(cols) * 3.2, len(SAMPLES) * 1.75))
        for si, (sc, st) in enumerate(SAMPLES):
            gt = np.load(f"{OUT}/gt_{si}.npy")
            for ci, (title, key) in enumerate(cols):
                ax = axes[si, ci]
                ax.set_xticks([]); ax.set_yticks([])
                if key == "gt":
                    img = gt
                else:
                    p = f"{OUT}/pred_{key}_{mode}_{si}.npy"
                    img = np.load(p) if os.path.exists(p) else None
                if img is None:                       # blank cell (byd / not-yet)
                    ax.set_facecolor("0.94")
                    for s in ax.spines.values():
                        s.set_visible(False)
                else:
                    ax.imshow(img, cmap="turbo", vmin=0, vmax=dm.MAX_DEPTH, aspect="auto")
                if si == 0:
                    ax.set_title(title, fontsize=11)
                if ci == 0:
                    ax.set_ylabel(f"{sc[:12]}\n#{st}", fontsize=8)
        fig.suptitle(f"Replica off3 test — {nch}-channel", fontsize=13, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fp = f"{OUT}/ch{nch}.png"
        fig.savefig(fp, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {fp}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compose", action="store_true")
    a = ap.parse_args()
    (compose if a.compose else predict)()
