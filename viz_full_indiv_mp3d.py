"""Individual native-resolution panels (one PNG per model per sample).

Output: comparison/viz_all_hires/ch{N}/indiv/{model}/{scene}_{step:03d}.png
model in {gt, rn, vit, es, bat, eco, oaa} — each file is the exact 256x512 depth map
(turbo, 0..10m), no text/borders. Complements the strips rendered by viz_full_hires.py.

  CUDA_VISIBLE_DEVICES=? REPLICA_ROOT=... DATA_MODULE=data_0422 python3 viz_full_indiv.py
"""
import os
os.environ.setdefault("DATA_MODULE", "data_mp3d")
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm
from PIL import Image

import data_mp3d as dm
import eval as ev

MAIN = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(MAIN, "comparison_mp3d", "viz_all")
ECO_DIR = os.path.join(MAIN, "comparison_mp3d", "viz_all")
MODES = {"r2": 2, "fb": 4, "r6": 6, "r8": 8}
CKPTS = {m: {"bat": f"comparison_mp3d/bat_{m}_fin",
             "oaa": f"comparison_mp3d/oaa_{m}_fin"} for m in MODES}
KEYS = ["rgb", "gt", "oaa", "eco", "bat"]
RGB_ROOT = os.environ.get("VIZ_RGB_ROOT", "")
TURBO = cm.get_cmap("turbo")

def save(depth, path):
    x = np.clip(depth / dm.MAX_DEPTH, 0, 1)
    Image.fromarray((TURBO(x)[..., :3] * 255).astype(np.uint8)).save(path)

device = torch.device("cuda")
samples = dm._index("test")

for mode, nch in MODES.items():
    if mode not in os.environ.get("VIZ_MODES", "r2,fb,r6,r8").split(","):
        continue
    dirs = {}
    for k in KEYS:
        d = os.path.join(OUT, f"ch{nch}", "indiv", k)
        os.makedirs(d, exist_ok=True); dirs[k] = d
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
                nm = f"{sc}_{st:03d}.png"
                rp = f"{RGB_ROOT or dm.ROOT}/{sc}/erp_rgb/erp_{st:03d}.png"
                if os.path.exists(rp) and not os.path.exists(os.path.join(dirs["rgb"], nm)):
                    Image.open(rp).convert("RGB").resize((512, 256), Image.BILINEAR).save(os.path.join(dirs["rgb"], nm))
                save(gt[j], os.path.join(dirs["gt"], nm))
                for key in ["bat", "oaa"]:
                    if key in preds:
                        save(preds[key][j], os.path.join(dirs[key], nm))
                if eco is not None:
                    save(np.asarray(eco[idx + j], dtype=np.float32), os.path.join(dirs["eco"], nm))
            idx += B
            if idx % 240 == 0:
                print(f"[ch{nch}] {idx}/{len(ds)}", flush=True)
    for net, *_ in nets.values():
        del net
    torch.cuda.empty_cache()
    print(f"[done] ch{nch}: {idx}x{len(KEYS)} PNGs -> {os.path.join(OUT, f'ch{nch}', 'indiv')}", flush=True)
print("[all done]", flush=True)
