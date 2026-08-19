"""Mic attribution for EchoDiffusion (isolated env): occlusion + keep-only maps.

Same protocol as viz/mic_attr.py (per-mic drop -> mean |D_full - D_drop|; per-mic solo -> error).
No attention readout: the channel-stacked stem mixes all mics at conv-1, so no mic tokens exist.
Front mics (ch 0/1) also zero their CIDE waveform channel when dropped.
Saves npy + verify json; PNG rendering is done separately in the main env.

  CUDA_VISIBLE_DEVICES=? DATA_MODULE=data_0422 R0422_SPLIT=off3 HF_HOME=... \
    <echodiff_env>/bin/python viz/mic_attr_eco.py --run-name eco_r8_fin
"""
# --- repo-root bootstrap: importable root modules (eval, data_*, model) + relative comparison/ paths
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
_os.chdir(ROOT)
import os, json, math, argparse
import numpy as np
import torch
from model.echodiffusion import EchoDiffusionDepth

from core.data import get_data_module
_DM = get_data_module("data_0422")
OUT = "comparison/mic_attribution"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="eco_r8_fin")
    ap.add_argument("--ckpt", default="best")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda")
    rd = os.path.join("comparison", a.run_name)
    ck = torch.load(os.path.join(rd, f"{a.ckpt}.pth"), map_location="cpu", weights_only=False)
    args = ck["args"]; mode = args.get("mode", "r8"); md = args.get("max_depth", 10.0)
    N = _DM.IN_CH[mode]
    wm = args.get("wave_mode", "std"); wch = N if wm == "all" else 2
    model = EchoDiffusionDepth(in_ch=N, wave_mode="none" if wm == "none" else "cide",
                               wave_ch=wch, faithful=(args.get("port", "enhanced") == "faithful")).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    ld = _DM.spec_wave_loader("test", 6, False, 5, mode)
    H, W = 256, 512
    occ = torch.zeros(N, H, W); keep_err = torch.zeros(N, H, W); keep_cnt = torch.zeros(N, H, W)
    nimg = 0
    for b in ld:
        x0 = b["spec"].to(device); w0 = b["wave"][:, :wch].to(device)
        gt = b["depth"].to(device) * md; mask = b["mask"].to(device)
        Df = model(x0, w0).float() * md
        for i in range(N):
            xd = x0.clone(); xd[:, i] = 0
            wd = w0
            if i < wch:
                wd = w0.clone(); wd[:, i] = 0
            xk = torch.zeros_like(x0); xk[:, i] = x0[:, i]
            wk = torch.zeros_like(w0)
            if i < wch:
                wk[:, i] = w0[:, i]
            Dd = model(xd, wd).float() * md
            Dk = model(xk, wk).float() * md
            occ[i] += (Df - Dd).abs().squeeze(1).sum(0).cpu()
            keep_err[i] += ((Dk - gt).abs() * mask).squeeze(1).sum(0).cpu()
            keep_cnt[i] += mask.squeeze(1).sum(0).cpu()
        nimg += x0.shape[0]
    occ /= nimg
    keep = (keep_err / keep_cnt.clamp(min=1))
    np.save(f"{OUT}/occlusion_{a.run_name}.npy", occ.numpy())
    np.save(f"{OUT}/keeponly_{a.run_name}.npy", keep.numpy())
    P = _DM.POSES[mode]
    def circ_peak(m):
        col = m.mean(0)
        th = np.linspace(0, 2 * np.pi, len(col), endpoint=False)
        z = (col * np.exp(1j * th)).sum() / max(col.sum(), 1e-9)
        return math.degrees(np.angle(z)) % 360, abs(z)
    rows = []
    for i, (yaw, ear) in enumerate(P):
        po, ro = circ_peak(occ[i].numpy())
        rows.append((i, math.degrees(yaw) % 360, "R" if ear > 0 else "L", po, ro,
                     float(occ[i].max() / occ[i].mean())))
    json.dump({"rows": rows, "note": "idx, yaw_deg, ear, occl_peak_az, occl_conc, occl_flatness"},
              open(f"{OUT}/verify_{a.run_name}.json", "w"), indent=2)
    print("idx yaw ear | occl_az conc flat")
    for r in rows:
        print(f"{r[0]}  {r[1]:5.0f} {r[2]} | {r[3]:6.1f} {r[4]:.2f} {r[5]:.1f}", flush=True)
    print(f"[saved] {OUT}/occlusion_{a.run_name}.npy etc.", flush=True)


if __name__ == "__main__":
    main()
