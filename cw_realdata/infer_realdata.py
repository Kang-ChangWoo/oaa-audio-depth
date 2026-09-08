#!/usr/bin/env python3
"""Real-recording inference with the campaign-best 8ch model (0820_sslam_llrd_r8vd_rep).

Input: cw_realdata/clipped_audio/mono_{45..360}deg_clip_1s.wav
       8 mono clips, 44.1 kHz, 1 s, one per rig azimuth (45-degree ring).

The model was trained on SoundSpaces Replica IRs with a very specific input contract
(data_0422.py); this script maps the real recordings onto that contract, making every
assumption explicit:

1) RIG MAPPING. Training "r8" is NOT an 8-mic ring: it is a binaural (L/R ear, +-y) pair
   at 4 yaws (0/90/180/270 deg), channel order [(0,L),(0,R),(90,L),(90,R),(180,L),(180,R),
   (270,L),(270,R)]. Assuming L ear sits at yaw+90 and R at yaw-90 (azimuth CCW, 0=front),
   the eight channels' nominal ear azimuths are [90,270,180,0,270,90,0,180] - four cardinal
   azimuths visited twice. We assign each channel the real mic closest to its nominal
   azimuth, using the 45-degree-offset mics for the duplicate visits so every real mic is
   used exactly once (each within 45 deg of nominal). The real rig's azimuth zero /
   handedness vs. the model frame is unknown, so the output panorama is defined only up to
   a global rotation (multiple of 45 deg) and a possible mirror - we save a mirrored
   (L/R-swapped) variant as well, since L/R handedness was a known ambiguity in training.
2) TIME ORIGIN. Training IRs have the direct spike at sample 49 (48 kHz) in every channel.
   Real clips are independently trimmed (peaks at 4-10 ms), so absolute sync is lost anyway;
   we place each channel's strongest peak (assumed direct) at sample 49, matching training.
3) RESAMPLING. 44.1 kHz -> 48 kHz polyphase (160/147), then cut WINDOW=2799 samples
   (10 m round trip - the model physically cannot see anything beyond ~10 m).
4) AMPLITUDE. Training normalises each stereo pair file so the louder ear peaks at 1.0
   (verified on train wavs); echo-to-direct amplitude ratio is a key cue (report S6/G).
   We therefore keep each channel's internal ratio and normalise per mapped yaw-pair.
5) SPECTROGRAM. Identical to data_0422._stft_mag: n_fft 512 / win 400 / hop 160, magnitude,
   nearest-resize to (256,512).

Outputs (this directory): pred_depth.npy / pred_depth_mirror.npy (metres, ERP HxW),
pred_depth.png (visualisation incl. both variants + input sanity panel).

Run from the repo root:
  REPLICA_ROOT=/root/local2/replica_0422_lite R0422_SPLIT=off3 DATA_MODULE=data_0422 \
  CUDA_VISIBLE_DEVICES=<gpu> python3 cw_realdata/infer_realdata.py
"""
import os, sys, glob, wave
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
os.environ.setdefault("REPLICA_ROOT", "/root/local2/replica_0422_lite")
os.environ.setdefault("R0422_SPLIT", "off3")
os.environ.setdefault("DATA_MODULE", "data_0422")

from scipy.signal import resample_poly
from core.data import get_data_module
from core.ckpt import build

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.path.dirname(HERE), "comparison_0820", "0820_sslam_llrd_r8vd_rep")
DIRECT_AT = 49          # training direct-spike sample index @48k

# channel order = data_0422 r8; nominal ear azimuth assuming L = yaw+90, R = yaw-90
# assignment: exact-azimuth mic on first visit, +45-offset mic on the duplicate visit
CH2MIC = {  # (yaw_slot, ear) -> real mic file azimuth ("360" = 0 deg)
    (0, "L"): 90,  (0, "R"): 270,
    (1, "L"): 180, (1, "R"): 360,
    (2, "L"): 315, (2, "R"): 135,
    (3, "L"): 45,  (3, "R"): 225,
}

def load_mono(deg):
    w = wave.open(os.path.join(HERE, "clipped_audio", f"mono_{deg}deg_clip_1s.wav"))
    assert w.getframerate() == 44100 and w.getnchannels() == 1 and w.getsampwidth() == 2
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return x

def to_training_window(x44):
    x48 = resample_poly(x44, 160, 147).astype(np.float32)          # 44.1k -> 48k
    pk = int(np.argmax(np.abs(x48)))                                # assume strongest peak = direct
    lead = DIRECT_AT
    start = pk - lead
    if start < 0:
        x48 = np.concatenate([np.zeros(-start, np.float32), x48]); start = 0
    seg = x48[start:start + 2799]
    if len(seg) < 2799:
        seg = np.pad(seg, (0, 2799 - len(seg)))
    return seg

def main():
    DM = get_data_module()
    WINDOW, stft = DM.WINDOW, DM._stft_mag
    assert WINDOW == 2799
    # ---- assemble 8 channels in r8 order, per-pair peak normalisation (training contract)
    order = [(o, e) for o in (0, 1, 2, 3) for e in ("L", "R")]
    chans = {ch: to_training_window(load_mono(CH2MIC[ch])) for ch in order}
    for o in (0, 1, 2, 3):                                          # per yaw-pair norm to peak 1.0
        pk = max(np.abs(chans[(o, "L")]).max(), np.abs(chans[(o, "R")]).max())
        chans[(o, "L")] /= pk; chans[(o, "R")] /= pk
    wav8 = torch.from_numpy(np.stack([chans[ch] for ch in order]))  # (8, 2799)
    spec = stft(wav8).unsqueeze(0)                                  # (1, 8, 256, 512)

    # ---- model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(RUN, "best.pth"), map_location="cpu", weights_only=False)
    model, dmode, nch, kind, poses = build(ck["args"], DM)
    assert dmode == "r8" and nch == 8 and kind == "spec"
    model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
    max_depth = ck["args"].get("max_depth", 10.0)

    with torch.no_grad():
        pred = (model(spec.to(dev), view_poses=poses).float() * max_depth).squeeze().cpu().numpy()
        # mirrored variant: swap L/R mic assignment within each pair (handedness unknown)
        idx = [1, 0, 3, 2, 5, 4, 7, 6]
        pred_m = (model(spec[:, idx].to(dev), view_poses=poses).float() * max_depth).squeeze().cpu().numpy()

    np.save(os.path.join(HERE, "pred_depth.npy"), pred)
    np.save(os.path.join(HERE, "pred_depth_mirror.npy"), pred_m)
    print("pred shape", pred.shape, "range %.2f..%.2f m mean %.2f" % (pred.min(), pred.max(), pred.mean()))
    print("mirror     range %.2f..%.2f m mean %.2f" % (pred_m.min(), pred_m.max(), pred_m.mean()))

    # ---- visualisation
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 8.5))
    ax = fig.add_subplot(2, 2, 1); im = ax.imshow(pred, cmap="turbo", vmin=0, vmax=max_depth, aspect="auto")
    ax.set_title("predicted ERP depth (m) — canonical mapping"); fig.colorbar(im, ax=ax, shrink=.8)
    ax.set_xlabel("azimuth (panorama x)"); ax.set_ylabel("elevation")
    ax = fig.add_subplot(2, 2, 2); im = ax.imshow(pred_m, cmap="turbo", vmin=0, vmax=max_depth, aspect="auto")
    ax.set_title("mirrored (L/R-swapped) variant"); fig.colorbar(im, ax=ax, shrink=.8)
    ax = fig.add_subplot(2, 2, 3)
    mid = pred[pred.shape[0]//2 - 2: pred.shape[0]//2 + 3].mean(0)
    midm = pred_m[pred_m.shape[0]//2 - 2: pred_m.shape[0]//2 + 3].mean(0)
    azx = np.linspace(0, 360, len(mid), endpoint=False)
    ax.plot(azx, mid, label="canonical"); ax.plot(azx, midm, label="mirrored", alpha=.7)
    ax.set_xlabel("azimuth (deg, model frame)"); ax.set_ylabel("horizon depth (m)"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("horizon profile (middle 5 rows)")
    ax = fig.add_subplot(2, 2, 4)
    t = np.arange(2799) / 48000 * 1000
    for k, ch in enumerate(order):
        ax.plot(t, wav8[k].numpy() * 0.8 + k, lw=.4)
    ax.set_yticks(range(8)); ax.set_yticklabels([f"y{o*90} {e} (mic{CH2MIC[(o,e)]})" for o, e in order], fontsize=7)
    ax.set_xlabel("time (ms)"); ax.set_title("model inputs after alignment/normalisation")
    fig.suptitle("cw_realdata → 0820_sslam_llrd_r8vd_rep (campaign-best 8ch) · panorama defined up to 45° rotation + mirror")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "pred_depth.png"), dpi=110)
    print("saved", os.path.join(HERE, "pred_depth.png"))

if __name__ == "__main__":
    main()
