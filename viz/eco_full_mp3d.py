"""EchoDiffusion predictions for the FULL test set (isolated env) -> one float16 memmap per mode.

  CUDA_VISIBLE_DEVICES=? REPLICA_ROOT=... <echodiff_env>/bin/python viz/eco_full.py
Writes comparison/viz_all/eco_{mode}.npy with shape (N_test, 256, 512); consumed by viz/full.py
and deleted after the PNGs are rendered.
"""
# --- repo-root bootstrap: importable root modules (eval, data_*, model) + relative comparison/ paths
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
_os.chdir(ROOT)
import os
os.environ.setdefault("DATA_MODULE", "data_mp3d")
import numpy as np
import torch
from torch.utils.data import DataLoader

import data_mp3d as dm
from model.echodiffusion import EchoDiffusionDepth

MAIN = ROOT
OUT = os.path.join(MAIN, "comparison_mp3d", "viz_all")
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda")

FIN = {"r2": "eco_r2_wstd", "fb": "eco_fb_wstd", "r6": "eco_r6", "r8": "eco_r8"}   # wstd fins
for mode in os.environ.get("VIZ_MODES", "r2,fb,r6,r8").split(","):
    f = os.path.join(MAIN, "comparison_mp3d", FIN[mode], "best.pth")
    if not os.path.exists(f):
        print(f"[skip] eco/{mode}"); continue
    ck = torch.load(f, map_location="cpu", weights_only=False)
    md = ck["args"].get("max_depth", 10.0)
    net = EchoDiffusionDepth(in_ch=dm.IN_CH[mode]).to(device)
    net.load_state_dict(ck["state_dict"]); net.eval()
    ds = dm.SpecWaveSet("test", mode)
    N = len(ds)
    mm = np.lib.format.open_memmap(f"{OUT}/eco_{mode}.npy", mode="w+", dtype=np.float16, shape=(N, 256, 512))
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=8)
    i = 0
    with torch.no_grad():
        for b in dl:
            D = net(b["spec"].to(device), b["wave"][:, :2].to(device)).float() * md
            n = D.shape[0]
            mm[i:i + n] = D[:, 0].cpu().numpy().astype(np.float16)
            i += n
    mm.flush(); del net; torch.cuda.empty_cache()
    print(f"[ok] eco/{mode} ({i} preds)", flush=True)
