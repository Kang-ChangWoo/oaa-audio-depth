"""Full-test-set qualitative strips: one PNG per test sample per channel mode.

Output: comparison/viz_all/ch{2,4,6,8}/{scene}_{step:03d}.png — a 1x8 strip
  GT | ResNet | ViT | Beyond(blank) | EchoScan | BatVision | EchoDiffusion | OAA(ours)
with the scene/step in the title. EchoDiffusion comes from the eco_{mode}.npy memmaps
written by viz_eco_full.py (isolated env); blank if absent. OAA 6/8ch uses the bmax
stand-in until the fe runs finish (swap CKPTS and re-run).

  CUDA_VISIBLE_DEVICES=? REPLICA_ROOT=... DATA_MODULE=data_0422 python3 viz_full.py
"""
import os
os.environ.setdefault("DATA_MODULE", "data_mp3d")
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import data_mp3d as dm
import eval as ev

MAIN = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(MAIN, "comparison_mp3d", "viz_all")
MODES = {"r2": 2, "fb": 4, "r6": 6, "r8": 8}
CKPTS = {m: {"bat": f"comparison_mp3d/bat_{m}_fin"} for m in MODES}
CKPTS["r2"]["oaa"] = "comparison_mp3d/oaa_r2_fin"
CKPTS["fb"]["oaa"] = "comparison_mp3d/oaa_fb_fin"
CKPTS["r6"]["oaa"] = "comparison_mp3d/oaa_r6_fin"
CKPTS["r8"]["oaa"] = "comparison_mp3d/oaa_r8_fin"
COLS = [("RGB", "rgb"), ("GT", "gt"), ("OAA (ours)", "oaa"), ("EchoDiffusion", "eco"), ("BatVision", "bat")]
RGB_ROOT = os.environ.get("VIZ_RGB_ROOT", "")          # "" -> use dm.ROOT (erp_rgb/erp_NNN.png)

device = torch.device("cuda")
samples = dm._index("test")                                # ordered (scene, step)

for mode, nch in MODES.items():
    if mode not in os.environ.get("VIZ_MODES", "r2,fb,r6,r8").split(","):
        continue
    ch_dir = os.path.join(OUT, f"ch{nch}")
    os.makedirs(ch_dir, exist_ok=True)
    # load this mode's models
    nets = {}
    for key, path in CKPTS[mode].items():
        f = os.path.join(MAIN, path, "best.pth")
        if not os.path.exists(f):
            print(f"[warn] no ckpt {path}", flush=True); continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        net, _dm_, _nch_, kind, poses = ev.build(ck["args"])
        net.load_state_dict(ck["state_dict"]); net.to(device).eval()
        nets[key] = (net, kind, poses, ck["args"].get("max_depth", 10.0))
    eco_f = f"{OUT}/eco_{mode}.npy"
    eco = np.load(eco_f, mmap_mode="r") if os.path.exists(eco_f) else None
    ds = dm.SpecWaveSet("test", mode)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=8)
    idx = 0
    with torch.no_grad():
        for b in dl:
            spec = b["spec"].to(device); wave = b["wave"].to(device)
            gt = (b["depth"][:, 0] * dm.MAX_DEPTH).numpy()
            preds = {}
            for key, (net, kind, poses, md) in nets.items():
                x = wave if kind == "wave" else spec
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    D = (net(x, view_poses=poses) if poses is not None else net(x)).float() * md
                preds[key] = D[:, 0].cpu().numpy()
            B = gt.shape[0]
            for j in range(B):
                sc, st = samples[idx + j]
                fig, axes = plt.subplots(1, len(COLS), figsize=(len(COLS) * 2.6, 1.75))
                for ci, (title, key) in enumerate(COLS):
                    ax = axes[ci]; ax.set_xticks([]); ax.set_yticks([])
                    if key == "rgb":
                        rp = f"{RGB_ROOT or dm.ROOT}/{sc}/erp_rgb/erp_{st:03d}.png"
                        img = np.asarray(Image.open(rp).convert("RGB")) if os.path.exists(rp) else None
                    else:
                        img = (gt[j] if key == "gt" else
                               preds.get(key)[j] if key in preds else
                               np.asarray(eco[idx + j], dtype=np.float32) if (key == "eco" and eco is not None) else None)
                    if img is None:
                        ax.set_facecolor("0.94")
                        for s in ax.spines.values():
                            s.set_visible(False)
                    elif key == "rgb":
                        ax.imshow(img, aspect="auto")
                    else:
                        ax.imshow(img, cmap="turbo", vmin=0, vmax=dm.MAX_DEPTH, aspect="auto")
                    ax.set_title(title, fontsize=8)
                fig.suptitle(f"{sc}  #{st:03d}  ({nch}ch)", fontsize=10, y=1.04)
                fig.tight_layout()
                fig.savefig(os.path.join(ch_dir, f"{sc}_{st:03d}.png"), dpi=80, bbox_inches="tight")
                plt.close(fig)
            idx += B
            if idx % 160 == 0:
                print(f"[ch{nch}] {idx}/{len(ds)}", flush=True)
    for net, *_ in nets.values():
        del net
    torch.cuda.empty_cache()
    print(f"[done] ch{nch}: {idx} PNGs -> {ch_dir}", flush=True)
print("[all done]", flush=True)
