"""E135 data module: data_0422 + EXPLICIT interaural input channels per ear observation.

Why this module exists (see PREREG_E135.md / the gate section of REPORT_E135.md):
  * The released front-end eats `torch.stft(...).abs()` -- interaural PHASE is discarded at the
    input, and the magnitude time axis cannot carry it either (hop 160 = 3.333 ms, so the maximum
    interaural time difference of ~0.66 ms is 0.198 of one frame).
  * OAAv2Depth encodes every ear observation INDEPENDENTLY through a weight-shared encoder
    (model/oaa.py `_encode`: spec.view(B, nv, in_ch, H, W).reshape(B*nv, in_ch, H, W)), so the two
    ears cannot be compared until the 16x32 token stage, i.e. after a 16x downsample on both axes.
  * OAAv2Depth already carries a per-observation channel axis `in_ch`, unused by train_oaa.py
    (always 1). This module fills it with the interaural comparison, so the ear-direction machinery
    OAA already has (ear-sign AdaLN conditioning, the ray-dot-ear-axis cross-attention bias) finally
    receives an interaural measurement instead of having to reconstruct one after downsampling.

Channel layout per ear e (other ear o), grouped per observation exactly as `_encode` expects
([obs0 ch0..chK-1, obs1 ch0..chK-1]):
  ch0  |S_e|                                        identical to the released input (ctrl is nested)
  ILD  (log10|S_e| - log10|S_o|) * SCALE_ILD        signed per ear: left sees +, right sees -
  IPDc cos(phi_e - phi_o) * SCALE_IPD               even in ear order (shared by both observations)
  IPDs sin(phi_e - phi_o) * SCALE_IPD               odd in ear order

Cue sets (env E135_CUE, K = channels per observation):
  none  K=1  [mag]                      byte-identical to data_0422 (retrain control)
  ild   K=2  [mag, ILD]                 interaural LEVEL only
  ipd   K=3  [mag, IPDc, IPDs]          interaural PHASE only
  bin4  K=4  [mag, ILD, IPDc, IPDs]     both

SCALE_* equalise the new channels' train-split std to the magnitude channel's (the OAA stem is a
bare Conv2d with no input normalisation). Constants measured once by e135_scale.py; see
results/cue_scale.json.

r2 (front binaural pair) only -- the multi-heading modes are not part of E135.
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import data_0422 as D

# ---- re-export the data-module contract that core/ckpt.py and the trainer read
SR, MAX_DEPTH, WINDOW = D.SR, D.MAX_DEPTH, D.WINDOW
N_FFT, WIN, HOP = D.N_FFT, D.WIN, D.HOP
H, W, ROOT = D.H, D.W, D.ROOT
MODES, IN_CH, POSES = D.MODES, D.IN_CH, D.POSES

CUE = os.environ.get("E135_CUE", "none")
CUE_CH = {"none": 1, "ild": 2, "ipd": 3, "bin4": 4}
assert CUE in CUE_CH, f"E135_CUE={CUE!r} not in {tuple(CUE_CH)}"
K = CUE_CH[CUE]

# Measured on 400 random train samples by e135_scale.py (results/cue_scale.json):
#   std(|S|) = 0.43050, std(ILD) = 0.40907, std(cos IPD) = 0.66471
# -> scale = std(mag)/std(cue) so every input channel has train std ~= 0.431.
# (cos IPD's train MEAN is +0.4249, not 0: the two ears are genuinely phase-coherent at a head-sized
#  spacing, i.e. this channel carries structure, not noise.)
SCALE_ILD = float(os.environ.get("E135_SCALE_ILD", "1.0524"))
SCALE_IPD = float(os.environ.get("E135_SCALE_IPD", "0.6476"))


def cue_channels(S, cue=None):
    """S complex (2, F, T) for [left, right] -> (2*K, F, T), grouped per observation."""
    cue = CUE if cue is None else cue
    mag = S.abs()
    if cue == "none":
        return mag
    lm = torch.log10(mag + 1e-8)
    ild = (lm[0] - lm[1]) * SCALE_ILD
    dphi = torch.angle(S[0] * torch.conj(S[1]))
    ipc = torch.cos(dphi) * SCALE_IPD
    ips = torch.sin(dphi) * SCALE_IPD
    out = []
    for e, sgn in ((0, 1.0), (1, -1.0)):          # ear 0 = left (+), ear 1 = right (-)
        ch = [mag[e]]
        if cue in ("ild", "bin4"):
            ch.append(sgn * ild)
        if cue in ("ipd", "bin4"):
            ch += [ipc, sgn * ips]
        out.append(torch.stack(ch))
    return torch.cat(out, 0)


def _stft_cue(wav2):
    """wav2 (2, WINDOW) -> (2*K, 256, 512); ch0 of each observation == data_0422._stft_mag exactly."""
    S = torch.stft(wav2, n_fft=N_FFT, hop_length=HOP, win_length=WIN, window=D._HANN,
                   center=True, return_complex=True)                       # (2, 257, T)
    x = cue_channels(S)
    return F.interpolate(x.unsqueeze(0), size=(H, W), mode="nearest")[0]    # same resize as data_0422


def _spec(scene, front, mode):
    assert mode == "r2", f"E135 is r2-only, got {mode}"
    step = D._group_steps(front, (0,))[0]
    return _stft_cue(D._load_wave1(f"{ROOT}/{scene}/audio_wav/audio_{step:03d}.wav"))


class RotSet(Dataset):
    """Binaural spec with interaural cue channels (2*K, 256, 512) + depth/mask."""

    def __init__(self, split, mode="r2"):
        assert mode == "r2", f"E135 is r2-only, got {mode}"
        self.mode = mode
        self.samples = D._index(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sc, front = self.samples[i]
        depth, mask = D._load_depth(f"{ROOT}/{sc}/erp_depth/erp_depth_{front:03d}.npy")
        return {"spec": _spec(sc, front, self.mode), "depth": depth, "mask": mask, "idx": i}


def loader(split, batch_size, shuffle, num_workers, mode="r2", *_ignore, **_kw):
    return DataLoader(RotSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)
