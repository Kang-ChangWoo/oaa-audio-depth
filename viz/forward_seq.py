"""Forward-motion sequence inference for the supplementary video/strip.

Data: $FORWARD_SEQ_ROOT/<scene>/seq_XXXX/
  audio_wav/step_XXX_rel{000,090,180,270}.wav  (binaural per relative yaw)
  erp_rgb/step_XXX.png, erp_depth_radial/step_XXX.npy, poses.json

Each step's 8 channels are assembled in the training order
[rel000 L,R | rel090 L,R | rel180 L,R | rel270 L,R] and fed to the final
8-channel model (comparison/oaa_r8_fin). Output: side-by-side [RGB | Pred]
PNGs at native resolution under test_for_audio_tof/supple/<scene>/<seq>/.

  DATA_MODULE=data_0422 CUDA_VISIBLE_DEVICES=? python viz/forward_seq.py [--gt]
"""
# --- repo-root bootstrap: importable root modules (eval, data_*, model) + relative comparison/ paths
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
_os.chdir(ROOT)
import os, sys, glob, argparse
os.environ.setdefault("DATA_MODULE", "data_0422")
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm
from PIL import Image, ImageDraw

import data_0422 as dm
from core.data import get_data_module
from core.ckpt import build, resolve_run
_DM = get_data_module()

SEQ_ROOT = os.environ.get("FORWARD_SEQ_ROOT", "data/replica_0422_forward_seq")
OUT_ROOT = os.environ.get("FORWARD_SEQ_OUT", "supple")
CKPT = "comparison/oaa_r8_fin/best.pth"
RELS = ["000", "090", "180", "270"]
TURBO = cm.get_cmap("turbo")


def wav8(step_dir_prefix):
    chans = []
    for rel in RELS:
        w, sr = sf.read(f"{step_dir_prefix}_rel{rel}.wav", dtype="float32")
        assert sr == dm.SR and w.ndim == 2 and w.shape[1] == 2, (sr, w.shape)
        w = w[:dm.WINDOW].T                                   # (2, WINDOW)
        if w.shape[1] < dm.WINDOW:
            w = np.pad(w, ((0, 0), (0, dm.WINDOW - w.shape[1])))
        chans.append(w)
    return np.concatenate(chans, 0)                            # (8, WINDOW)


def spec8(w8, device):
    t = torch.from_numpy(w8).to(device)
    s = torch.stft(t, n_fft=512, win_length=400, hop_length=160,
                   window=torch.hann_window(400, device=device),
                   center=True, return_complex=True).abs()     # (8, 257, T)
    s = s[:, :256]
    s = F.interpolate(s.unsqueeze(0), size=(256, 512), mode="nearest")
    return s                                                    # (1, 8, 256, 512)


def to_rgb(depth_m):
    x = np.clip(depth_m / dm.MAX_DEPTH, 0, 1)
    return (TURBO(x)[..., :3] * 255).astype(np.uint8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", action="store_true", help="add a GT panel (RGB|GT|Pred)")
    a = ap.parse_args()
    device = torch.device("cuda")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model, dmode, nch, kind, poses = build(ck["args"], _DM)
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    md = ck["args"].get("max_depth", 10.0)
    n_out = 0
    for scene in sorted(os.listdir(SEQ_ROOT)):
        for seq in sorted(os.listdir(os.path.join(SEQ_ROOT, scene))):
            sdir = os.path.join(SEQ_ROOT, scene, seq)
            if not os.path.isdir(os.path.join(sdir, "audio_wav")):
                continue
            odir = os.path.join(OUT_ROOT, scene, seq)
            os.makedirs(odir, exist_ok=True)
            steps = sorted({os.path.basename(f).split("_rel")[0]
                            for f in glob.glob(f"{sdir}/audio_wav/*_rel000.wav")})
            for st in steps:
                op = os.path.join(odir, f"{st}.png")
                rp = f"{sdir}/erp_rgb/{st}.png"
                if os.path.exists(op) or not os.path.exists(rp):
                    continue                                   # skip already-rendered / missing-data steps
                x = spec8(wav8(f"{sdir}/audio_wav/{st}"), device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    D = model(x, view_poses=poses).float() * md
                pred = D[0, 0].cpu().numpy()
                rgb = np.asarray(Image.open(f"{sdir}/erp_rgb/{st}.png").convert("RGB"))
                if rgb.shape[:2] != (256, 512):
                    rgb = np.asarray(Image.fromarray(rgb).resize((512, 256), Image.BILINEAR))
                panels = [rgb]
                if a.gt:
                    gt = np.load(f"{sdir}/erp_depth_radial/{st}.npy").astype(np.float32)
                    if gt.shape != (256, 512):
                        gt = np.asarray(Image.fromarray(gt).resize((512, 256), Image.NEAREST))
                    panels.append(to_rgb(gt))
                panels.append(to_rgb(pred))
                SEP = 4
                W = sum(p.shape[1] for p in panels) + SEP * (len(panels) - 1)
                canvas = np.full((256, W, 3), 255, np.uint8)
                x0 = 0
                for p in panels:
                    canvas[:, x0:x0 + p.shape[1]] = p
                    x0 += p.shape[1] + SEP
                Image.fromarray(canvas).save(op)
                n_out += 1
            print(f"[{scene}/{seq}] {len(steps)} steps", flush=True)
    print(f"[done] {n_out} PNGs -> {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
