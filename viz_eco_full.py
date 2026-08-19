"""EchoDiffusion predictions for the FULL test set (isolated env) -> one float16 memmap per mode.

  CUDA_VISIBLE_DEVICES=? REPLICA_ROOT=... <echodiff_env>/bin/python viz_eco_full.py
Writes comparison/viz_all/eco_{mode}.npy with shape (N_test, 256, 512); consumed by viz_full.py
and deleted after the PNGs are rendered.
"""
import os
os.environ.setdefault("DATA_MODULE", "data_0422")
import numpy as np
import torch
from torch.utils.data import DataLoader

import data_0422 as dm
dm.ROOT = os.environ.get("REPLICA_ROOT", dm.ROOT)
from model.echodiffusion import EchoDiffusionDepth

MAIN = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(MAIN, "comparison", "viz_all")
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda")

for mode in os.environ.get("VIZ_MODES", "r2,fb,r6,r8").split(","):
    f = os.path.join(MAIN, f"comparison/eco_{mode}_fin/best.pth")
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
