"""EchoDiffusion(wstd fin) predictions for the MP3D viz_all grid — ISOLATED env (torch 1.13/ldm).

  CUDA_VISIBLE_DEVICES=? DATA_MODULE=data_mp3d HF_HOME=<hf_cache> \
    <echodiff_env>/bin/python viz_eco_mp3d.py
Saves comparison_mp3d/viz_all/pred_eco_{mode}_{si}.npy; viz_all_mp3d.py --compose picks them up.
"""
import os
os.environ.setdefault("DATA_MODULE", "data_mp3d")
import numpy as np
import torch

import data_mp3d as dm
from model.echodiffusion import EchoDiffusionDepth

MAIN = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(MAIN, "comparison_mp3d", "viz_all")
os.makedirs(OUT, exist_ok=True)
SAMPLES = [("8WUmhLawc2A", 0), ("8WUmhLawc2A", 220),
           ("EDJbREhghzL", 0), ("EDJbREhghzL", 220),
           ("HxpKQynjfin", 0), ("HxpKQynjfin", 220)]
FIN = {"r2": "eco_r2_wstd", "fb": "eco_fb_wstd", "r6": "eco_r6", "r8": "eco_r8"}   # wstd fins

device = torch.device("cuda")
for mode, run in FIN.items():
    f = os.path.join(MAIN, "comparison_mp3d", run, "best.pth")
    if not os.path.exists(f):
        print(f"[skip] eco/{mode}"); continue
    ck = torch.load(f, map_location="cpu", weights_only=False)
    a = ck["args"]; md = a.get("max_depth", 10.0)
    wm = a.get("wave_mode", "std")
    net = EchoDiffusionDepth(in_ch=dm.IN_CH[mode], wave_mode="none" if wm == "none" else "cide",
                             wave_ch=dm.IN_CH[mode] if wm == "all" else 2,
                             faithful=(a.get("port", "enhanced") == "faithful")).to(device)
    net.load_state_dict(ck["state_dict"]); net.eval()
    with torch.no_grad():
        for si, (sc, st) in enumerate(SAMPLES):
            wave = dm._load_wave(sc, st, mode)
            spec = dm._stft_mag(wave)
            D = net(spec.unsqueeze(0).to(device), wave[:2].unsqueeze(0).to(device)).float() * md
            np.save(f"{OUT}/pred_eco_{mode}_{si}.npy", D[0, 0].cpu().numpy())
    del net; torch.cuda.empty_cache()
    print(f"[ok] eco/{mode} ({run})", flush=True)
