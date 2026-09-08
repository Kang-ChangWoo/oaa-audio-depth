#!/usr/bin/env python3
"""EchoDiffusion real-recording inference (isolated env: run with $ECHODIFF_PY).

Same real->training-contract mapping as infer_realdata.py (see its docstring): channel
mapping to the binaural-x-4-yaw r8 rig, direct-at-49 alignment, 44.1->48k resample,
per-yaw-pair peak normalisation, data-module _stft_mag. EchoDiffusion additionally takes
the raw front-pair waveform (wave_mode=std -> first 2 channels) for its frozen wav2vec2
CIDE branch.

  ECHODIFF_PY=... (from env.local.sh)
  $ECHODIFF_PY cw_realdata/infer_realdata_eco.py --run comparison/eco_r8_fin --data-module data_0422 --tag room_replica_eco --out-dir results
"""
import os, sys, wave, argparse
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="checkpoint dir relative to repo root (comparison/eco_r8_fin | comparison_mp3d/eco_r8)")
ap.add_argument("--data-module", default="data_0422")
ap.add_argument("--wav48", required=True, help="precomputed aligned 48k (8,N) npy from results/_wav48_<scene>.npy")
ap.add_argument("--out-dir", default="results")
ap.add_argument("--tag", required=True)
ARGS = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.environ.setdefault("REPLICA_ROOT", "/root/local2/replica_0422_lite")
os.environ.setdefault("MP3D_ROOT", "/root/local1/changwoo/matterport3d_0303renew")
os.environ.setdefault("R0422_SPLIT", "off3")
os.environ.setdefault("HF_HOME", "/root/local1/changwoo/_afm_weights")
os.environ["DATA_MODULE"] = ARGS.data_module

from core.data import get_data_module
from model.echodiffusion import EchoDiffusionDepth

DIRECT_AT = 49
CH2MIC = {(0,"L"):90,(0,"R"):270,(1,"L"):180,(1,"R"):360,(2,"L"):315,(2,"R"):135,(3,"L"):45,(3,"R"):225}

DM = get_data_module()
w48 = np.load(os.path.join(HERE, ARGS.wav48))[:, :DM.WINDOW].copy()    # aligned, direct@49
for o in range(4):                                                      # per yaw-pair peak norm
    pk = max(np.abs(w48[2*o]).max(), np.abs(w48[2*o+1]).max())
    w48[2*o] /= pk; w48[2*o+1] /= pk
wav8 = torch.from_numpy(w48)
spec = DM._stft_mag(wav8).unsqueeze(0)

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(os.path.join(REPO, ARGS.run, "best.pth"), map_location="cpu", weights_only=False)
a = ck["args"]; md = a.get("max_depth",10.0); wm = a.get("wave_mode","std")
wch = 8 if wm == "all" else 2
model = EchoDiffusionDepth(in_ch=8, wave_mode="none" if wm=="none" else "cide", wave_ch=wch,
                           faithful=(a.get("port","enhanced")=="faithful")).to(dev)
model.load_state_dict(ck["state_dict"]); model.eval()
w2 = wav8[:wch].unsqueeze(0)
with torch.no_grad():
    pred = (model(spec.to(dev), w2.to(dev)).float()*md).squeeze().cpu().numpy()
    idx = [1,0,3,2,5,4,7,6]
    pred_m = (model(spec[:,idx].to(dev), wav8[idx][:wch].unsqueeze(0).to(dev)).float()*md).squeeze().cpu().numpy()

OUT = os.path.join(HERE, ARGS.out_dir); os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, f"pred_depth_{ARGS.tag}.npy"), pred)
np.save(os.path.join(OUT, f"pred_depth_mirror_{ARGS.tag}.npy"), pred_m)
print(ARGS.tag, "pred", pred.shape, "range %.2f..%.2f mean %.2f"%(pred.min(),pred.max(),pred.mean()))

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)   # env has no matplotlib; render pngs with the main python instead
fig,axes=plt.subplots(1,2,figsize=(13,3.6))
for ax,d,t in [(axes[0],pred,"canonical"),(axes[1],pred_m,"mirrored")]:
    im=ax.imshow(d,cmap="turbo",vmin=0,vmax=md,aspect="auto"); ax.set_title(f"{ARGS.tag} — {t}"); fig.colorbar(im,ax=ax,shrink=.85)
fig.tight_layout(); fig.savefig(os.path.join(OUT, f"pred_depth_{ARGS.tag}.png"), dpi=110)
print("saved", os.path.join(OUT, f"pred_depth_{ARGS.tag}.png"))
