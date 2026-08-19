"""EchoDiffusion predictions for the viz_all grid — run in the ISOLATED env (torch 1.13/ldm).

  CUDA_VISIBLE_DEVICES=? REPLICA_ROOT=... DATA_MODULE=data_0422 HF_HOME=<hf_cache> \
    <echodiff_env>/bin/python viz_eco.py
Saves comparison/viz_all/pred_eco_{mode}_{si}.npy; viz_all.py --compose picks them up.
"""
import os
os.environ.setdefault("DATA_MODULE", "data_0422")
import numpy as np
import torch

import data_0422 as dm
dm.ROOT = os.environ.get("REPLICA_ROOT", dm.ROOT)
from model.echodiffusion import EchoDiffusionDepth

MAIN = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(MAIN, "comparison", "viz_all")
os.makedirs(OUT, exist_ok=True)
SAMPLES = [("apartment_2", 0), ("apartment_2", 220),
           ("frl_apartment_5", 0), ("frl_apartment_5", 220),
           ("office_4", 0), ("office_4", 220)]

device = torch.device("cuda")
for mode in ["r2", "fb", "r6", "r8"]:
    f = os.path.join(MAIN, f"comparison/eco_{mode}_fin/best.pth")
    if not os.path.exists(f):
        print(f"[skip] eco/{mode}"); continue
    ck = torch.load(f, map_location="cpu", weights_only=False)
    md = ck["args"].get("max_depth", 10.0)
    net = EchoDiffusionDepth(in_ch=dm.IN_CH[mode]).to(device)
    net.load_state_dict(ck["state_dict"]); net.eval()
    with torch.no_grad():
        for si, (sc, st) in enumerate(SAMPLES):
            wave = dm._load_wave(sc, st, mode)
            spec = dm._stft_mag(wave)
            D = net(spec.unsqueeze(0).to(device), wave[:2].unsqueeze(0).to(device)).float() * md
            np.save(f"{OUT}/pred_eco_{mode}_{si}.npy", D[0, 0].cpu().numpy())
    del net; torch.cuda.empty_cache()
    print(f"[ok] eco/{mode}", flush=True)
