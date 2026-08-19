"""Gradient-based mic attribution for EchoDiffusion — the mechanistic stand-in for attention.

EchoDiffusion has no mic tokens (channels fuse at conv-1), so mic-level attention does not
exist. Substitute: for each ERP azimuth sector, |grad x input| of the sector's mean output
w.r.t. each input spectrogram channel -> (sector x mic) reference matrix, comparable to
OAA's RayMicAttn maps aggregated over the same sectors.

  CUDA_VISIBLE_DEVICES=? DATA_MODULE=data_0422 R0422_SPLIT=off3 HF_HOME=... \
    <echodiff_env>/bin/python viz_gradattr_eco.py --run-name eco_r8_fin
"""
import os, json, math, argparse, importlib
import numpy as np
import torch
from model.echodiffusion import EchoDiffusionDepth

_DM = importlib.import_module(os.environ.get("DATA_MODULE", "data_0422"))
OUT = "comparison/mic_attribution"
SECTORS = 16          # azimuth sectors (512/16 = 32-col blocks)
MAX_BATCHES = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="eco_r8_fin")
    ap.add_argument("--ckpt", default="best")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda")
    rd = os.path.join("comparison", a.run_name)
    ck = torch.load(os.path.join(rd, f"{a.ckpt}.pth"), map_location="cpu", weights_only=False)
    args = ck["args"]; mode = args.get("mode", "r8")
    N = _DM.IN_CH[mode]
    wm = args.get("wave_mode", "std"); wch = N if wm == "all" else 2
    model = EchoDiffusionDepth(in_ch=N, wave_mode="none" if wm == "none" else "cide",
                               wave_ch=wch, faithful=(args.get("port", "enhanced") == "faithful")).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ld = _DM.spec_wave_loader("test", 4, False, 4, mode)
    W = 512; sw = W // SECTORS
    mat = np.zeros((SECTORS, N))
    nb = 0
    for bi, b in enumerate(ld):
        if bi >= MAX_BATCHES:
            break
        for s in range(SECTORS):
            x = b["spec"].to(device).requires_grad_(True)
            wv = b["wave"][:, :wch].to(device)
            D = model(x, wv)
            loss = D[..., s * sw:(s + 1) * sw].mean()
            g = torch.autograd.grad(loss, x)[0]
            mat[s] += (g * x.detach()).abs().sum(dim=(0, 2, 3)).cpu().numpy()
        nb += 1
    mat /= nb
    mat = mat / mat.sum(1, keepdims=True)          # normalise per sector (like attention mass)
    np.save(f"{OUT}/gradattr_{a.run_name}.npy", mat)

    P = _DM.POSES[mode]
    lbl = [f"{int(math.degrees(y))%360}°{'R' if e>0 else 'L'}" for y, e in P]
    rows = []
    for i in range(N):
        col = mat[:, i]
        th = (np.arange(SECTORS) + 0.5) / SECTORS * 2 * np.pi
        z = (col * np.exp(1j * th)).sum() / max(col.sum(), 1e-9)
        rows.append((i, math.degrees(P[i][0]) % 360, lbl[i][-1],
                     math.degrees(np.angle(z)) % 360, abs(z), float(col.max() / col.mean())))
    json.dump({"rows": rows, "note": "idx,yaw,ear,grad_peak_colangle,conc,flatness (az=colangle-180)"},
              open(f"{OUT}/verify_gradattr_{a.run_name}.json", "w"), indent=2)
    print("idx yaw ear | grad_az conc flat")
    for r in rows:
        print(f"{r[0]}  {r[1]:5.0f} {r[2]} | {r[3]:6.1f} {r[4]:.2f} {r[5]:.1f}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(mat.T, cmap="magma", aspect="auto")
    ax.set_yticks(range(N)); ax.set_yticklabels(lbl)
    ax.set_xlabel(f"ERP azimuth sector ({SECTORS} bins, left→right = col-angle 0→360°)")
    ax.set_title(f"{a.run_name} — grad×input mic reference per azimuth sector")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(f"{OUT}/gradattr_{a.run_name}.png", dpi=110); plt.close(fig)
    print(f"[saved] {OUT}/gradattr_{a.run_name}.png / .npy", flush=True)


if __name__ == "__main__":
    main()
