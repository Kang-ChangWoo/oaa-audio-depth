"""Native-resolution qualitative strips (no matplotlib figure resampling).

Output: comparison/viz_all_hires/ch{2,4,6,8}/{scene}_{step:03d}.png
Each panel is the exact 256x512 prediction (turbo, 0..10m) concatenated horizontally
with 4px white separators, in the paper column order:
  GT | ResNet | ViT | Beyond(blank) | EchoScan | BatVision | EchoDiffusion | OAA(ours)
A small header strip carries the column labels. EchoDiffusion panels read the
eco_{mode}.npy memmaps under viz_all/ (blank if absent). CKPTS follow the *_fin runs
(2026-07-27: oaa_r8_fin = vdrop+wu8 rounds2 champion).

  CUDA_VISIBLE_DEVICES=? REPLICA_ROOT=... DATA_MODULE=data_0422 python3 viz_full_hires.py
"""
import os
os.environ.setdefault("DATA_MODULE", "data_0422")
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageDraw

import data_0422 as dm
dm.ROOT = os.environ.get("REPLICA_ROOT", dm.ROOT)
import eval as ev

MAIN = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(MAIN, "comparison", "viz_all")
ECO_DIR = os.path.join(MAIN, "comparison", "viz_all")          # eco memmaps live here
MODES = {"r2": 2, "fb": 4, "r6": 6, "r8": 8}
CKPTS = {m: {"rn": f"comparison/rn_{m}_fin", "vit": f"comparison/vit_{m}_fin",
             "es": f"comparison/es_{m}_fin", "bat": f"comparison/bat_{m}_fin",
             "oaa": f"comparison/oaa_{m}_fin"} for m in MODES}
COLS = [("GT", "gt"), ("ResNet-50", "rn"), ("ViT-B/16", "vit"), ("Beyond", "byd"),
        ("EchoScan", "es"), ("BatVision", "bat"), ("EchoDiffusion", "eco"), ("OAA (ours)", "oaa")]
H, W, SEP, HDR = 256, 512, 4, 22
TURBO = cm.get_cmap("turbo")

def to_rgb(depth):
    x = np.clip(depth / dm.MAX_DEPTH, 0, 1)
    return (TURBO(x)[..., :3] * 255).astype(np.uint8)

device = torch.device("cuda")
samples = dm._index("test")

for mode, nch in MODES.items():
    if mode not in os.environ.get("VIZ_MODES", "r2,fb,r6,r8").split(","):
        continue
    ch_dir = os.path.join(OUT, f"ch{nch}")
    os.makedirs(ch_dir, exist_ok=True)
    nets = {}
    for key, path in CKPTS[mode].items():
        f = os.path.join(MAIN, path, "best.pth")
        if not os.path.exists(f):
            print(f"[warn] no ckpt {path}", flush=True); continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        net, _d, _n, kind, poses = ev.build(ck["args"])
        net.load_state_dict(ck["state_dict"]); net.to(device).eval()
        nets[key] = (net, kind, poses, ck["args"].get("max_depth", 10.0))
    eco_f = os.path.join(ECO_DIR, f"eco_{mode}.npy")
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
                total_w = len(COLS) * W + (len(COLS) - 1) * SEP
                canvas = np.full((HDR + H, total_w, 3), 255, np.uint8)
                for ci, (title, key) in enumerate(COLS):
                    x0 = ci * (W + SEP)
                    img = (gt[j] if key == "gt" else
                           preds[key][j] if key in preds else
                           np.asarray(eco[idx + j], dtype=np.float32) if (key == "eco" and eco is not None) else None)
                    canvas[HDR:, x0:x0 + W] = to_rgb(img) if img is not None else 240
                im = Image.fromarray(canvas)
                dr = ImageDraw.Draw(im)
                for ci, (title, _k) in enumerate(COLS):
                    dr.text((ci * (W + SEP) + 6, 4), f"{title}   {sc} #{st:03d} ({nch}ch)" if ci == 0 else title, fill=(0, 0, 0))
                im.save(os.path.join(ch_dir, f"{sc}_{st:03d}.png"))
            idx += B
            if idx % 160 == 0:
                print(f"[ch{nch}] {idx}/{len(ds)}", flush=True)
    for net, *_ in nets.values():
        del net
    torch.cuda.empty_cache()
    print(f"[done] ch{nch}: {idx} PNGs -> {ch_dir}", flush=True)
print("[all done]", flush=True)
