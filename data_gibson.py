"""Gibson data module — byte-for-byte the data_mp3d pipeline on the Gibson render.

Selected at runtime via DATA_MODULE=data_gibson. Modes / STFT recipe / pose convention / window /
normalisation are copied verbatim from data_mp3d, so models and checkpoints transfer unchanged.
ONLY the data root, the split directory and the on-disk FILENAME convention differ.

Dataset: $GIBSON_ROOT/<scene>/{audio_wav,erp_depth_radial}/*  (2026-10-22 render, 400 poses per
scene = 100 locations x 4 yaws). Gibson names files by (location, yaw degrees):
    audio_LLL_YYY.wav / erp_depth_LLL_YYY.npy   with YYY in {000,090,180,270}
while MP3D uses a flat index: audio_NNN.wav. The flat index NNN maps to
    location = NNN // 4 , yaw = (NNN % 4) * 90
which reproduces MP3D's "poses in groups of 4 yaws" grouping exactly, so _group_steps and every
downstream index rule stay untouched.

Split: scene-disjoint 72/9/9 drawn from the Gibson official train(412) with a fixed seed, read from
splits/gibson/{split}_keys.json ("scene/idx"). Sized to match the MP3D protocol so per-cell epoch
budget is identical and the Gibson column is directly comparable to the MP3D column.
"""
import os, json, glob, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = os.environ.get("GIBSON_ROOT", "/root/local1/gibson_hear360")# set GIBSON_ROOT to the dataset root
KEYS = os.environ.get("GIBSON_KEYS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits", "gibson"))  # {split}_keys.json
SR = 48000
MAX_DEPTH = 10.0
# Input crop length is env-overridable (STFT_WINDOW, in samples) for the context ablation. The
# source wavs are far longer than the released crop -- Replica 1641 ms, MP3D >=1147 ms -- so a longer
# context needs no re-rendering. 210 ms (10080) at hop 160 yields exactly 64 STFT frames, matching
# the released crop at hop 44, which is what makes "denser sampling" separable from "longer acoustic
# support". A non-default value disables the spec cache and is recorded in the checkpoint args.
WINDOW = int(os.environ.get("STFT_WINDOW", 2823))   # MP3D legacy convention (340 m/s round-trip @10 m) — 2026-07-24: MP3D and Replica keep
#               # their own dataset-specific windows (Replica: 343 m/s = 2799). Matches cached checkpoints.
H, W = 256, 512
# STFT hop is env-overridable (STFT_HOP) for the input-resolution study: the released recipe
# (hop 160 = 3.33 ms @ 48 kHz) yields only ~18 real frames over the 59 ms clip, which the cache then
# nearest-upsamples to 512 columns -- a 28x replicated time axis. A smaller hop puts real information
# on the axis that encodes distance. Any value other than the default disables the pre-built spec
# cache (it was rendered at hop 160) and is recorded in the checkpoint args so eval can refuse a
# mismatch. Frequency content is untouched: only the sampling of the time axis changes.
# N_FFT / WIN are env-overridable (STFT_NFFT / STFT_WIN) alongside STFT_HOP, for the
# input-resolution study. They are NOT the same axis:
#   hop  = how densely the time axis is SAMPLED. The released 160 already satisfies hop <= win/2,
#          so 44 is ~4.5x oversampling and adds little genuinely new information.
#   win  = what the time axis can RESOLVE. win 400 (8.33 ms) is a 143 cm round-trip distance
#          resolution -- coarse for a task whose errors are ~25 cm. Shortening the window is the
#          only way to actually sharpen time; a smaller n_fft additionally cuts the frequency bin
#          count, so vary WIN alone (zero-padded STFT) to keep the two axes separable.
# Any non-default value disables the prebuilt spec cache and is recorded in the checkpoint args.
N_FFT = int(os.environ.get("STFT_NFFT", 512))
WIN   = int(os.environ.get("STFT_WIN", 400))
HOP   = int(os.environ.get("STFT_HOP", 160))

_OFFS = {"r2": (0,), "fb": (0, 2), "fs": (0, 1), "r6": (0, 1, 3), "r8": (0, 1, 2, 3)}  # group-relative yaw slots (L,R each)
# Channel-level specs (mirror of data_0422): (yaw-slot, ear) with ear 0=L, 1=R. cb = the MP3D
# champion config [0L, 0R, 90R, 270L] so cache-trained cB checkpoints evaluate natively.
_CH = {m: [(o, e) for o in offs for e in (0, 1)] for m, offs in _OFFS.items()}
_CH["cb"] = [(0, 0), (0, 1), (1, 1), (3, 0)]
# extra 4ch combinations (2026-07-28): ss=left/right side pairs, fr=front+right pairs, al/ar=4 directions single ear (all L / all R)
_CH["ss"] = [(1, 0), (1, 1), (3, 0), (3, 1)]
_CH["fr"] = [(0, 0), (0, 1), (3, 0), (3, 1)]
_CH["al"] = [(0, 0), (1, 0), (2, 0), (3, 0)]
_CH["ar"] = [(0, 1), (1, 1), (2, 1), (3, 1)]
_CH["xa"] = [(0, 0), (1, 1), (2, 0), (3, 1)]   # cross-diagonal: 0L,90R,180L,270R (4 directions + alternating ear)
_CH["cbm"] = [(0, 0), (0, 1), (1, 0), (3, 1)]  # mirror of cb: 0L,0R,90L,270R
_CH["cB"] = _CH["cb"]        # eval.py _NV2MODE fallback uses the cache module's capitalisation
MODES = tuple(_CH)
IN_CH = {m: len(ch) for m, ch in _CH.items()}
POSES = {m: [(o * (math.pi / 2), (-1.0, 1.0)[e]) for o, e in ch] for m, ch in _CH.items()}


_NSCENE = {}


def _stem(step):
    """MP3D flat index -> Gibson (location, yaw) stem. The ONLY on-disk naming difference."""
    return f"{step // 4:03d}_{(step % 4) * 90:03d}"


def _scene_n(sc):
    if sc not in _NSCENE:
        _NSCENE[sc] = len(glob.glob(f"{ROOT}/{sc}/audio_wav/audio_*.wav"))
    return _NSCENE[sc]


def _index(split):
    """Original MP3D split samples as (scene, front_step) from the cached key list.
    Samples whose 4-yaw group is incomplete on disk are dropped (ZMojNkEp431's render stops at
    audio_342 while the key list expects 343 — 3 val samples; multi-view modes crashed on it)."""
    keys = json.load(open(f"{KEYS}/{split}_keys.json"))
    out = []
    for k in keys:
        sc, idx = k.split("/")
        i = int(idx)
        if 4 * (i // 4) + 3 < _scene_n(sc):
            out.append((sc, i))
    return out


def _group_steps(front, offs):
    base, k = 4 * (front // 4), front % 4
    return [base + (k + o) % 4 for o in offs]


_HANN = torch.hann_window(WIN)


def _stft_mag(wav2):
    s = torch.stft(wav2, n_fft=N_FFT, hop_length=HOP, win_length=WIN, window=_HANN,
                   center=True, return_complex=True).abs()
    return F.interpolate(s.unsqueeze(0), size=(H, W), mode="nearest")[0]   # nearest = MP3D cache recipe


def _load_wave1(wpath, frames=WINDOW):
    import soundfile as sf
    y, _ = sf.read(wpath, frames=frames, dtype="float32", always_2d=True)
    if y.shape[0] < frames:
        y = np.pad(y, ((0, frames - y.shape[0]), (0, 0)))
    return torch.from_numpy(y.T.copy())


def _load_wave(scene, front, mode, frames=WINDOW):
    chans = _CH[mode]
    offs = sorted({o for o, _ in chans})
    step = dict(zip(offs, _group_steps(front, offs)))
    wav = {o: _load_wave1(f"{ROOT}/{scene}/audio_wav/audio_{_stem(step[o])}.wav", frames) for o in offs}
    return torch.cat([wav[o][e:e + 1] for o, e in chans], 0)


_SPEC_CACHE = os.environ.get("GIBSON_SPEC_CACHE", "") if (HOP, WIN, N_FFT, WINDOW) == (160, 400, 512, 2823) else ""   # output of tools/build_spec_cache_mp3d.py (fp32, bit-identical to on-the-fly)


def _spec1(scene, step):
    """Binaural spec (2,256,512) for one step: load from cache if present, else compute on the fly."""
    if _SPEC_CACHE:
        p = f"{_SPEC_CACHE}/{scene}/spec_{step:03d}.npy"
        if os.path.exists(p):
            return torch.from_numpy(np.load(p))
    return _stft_mag(_load_wave1(f"{ROOT}/{scene}/audio_wav/audio_{_stem(step)}.wav"))


def _spec(scene, front, mode):
    """Assemble step caches in the mode's channel order (identical to _stft_mag(_load_wave(...)))."""
    chans = _CH[mode]
    offs = sorted({o for o, _ in chans})
    step = dict(zip(offs, _group_steps(front, offs)))
    sp = {o: _spec1(scene, step[o]) for o in offs}
    return torch.cat([sp[o][e:e + 1] for o, e in chans], 0)


# ---- --pair-feat ipd (2026-11-01): interaural PHASE planes. The magnitude spec above discards phase,
# so these are computed from the waveform with the SAME STFT and the SAME nearest resize, following
# E135 (/root/local1/changwoo/e135/data_0422_bin.py cue_channels): dphi = angle(S_L * conj(S_R)) on
# the raw (257, T) STFT, planes cos(dphi)*IPD_SCALE and sin(dphi)*IPD_SCALE, resized AFTER cos/sin.
# Per channel the sin plane is ear-signed (+ for L, - for R) so each observation sees phi_own-phi_other.
# IPD_SCALE equalises the plane std to the magnitude's; the E135 value (0.6476, Replica train) is
# reused as-is via the same env knob E135_SCALE_IPD -- NOT re-measured on Gibson. EMIT_IPD defaults
# to False: the trainer / evaluator set it for pair_feat == "ipd"; every other path returns the
# plain batch dict, unchanged. The phase planes are never cached (the spec cache holds magnitudes).
IPD_SCALE = float(os.environ.get("E135_SCALE_IPD", "0.6476"))
EMIT_IPD = False


def _stft_ipd1(wav2):
    S = torch.stft(wav2, n_fft=N_FFT, hop_length=HOP, win_length=WIN, window=_HANN,
                   center=True, return_complex=True)                      # (2, 257, T) complex, L then R
    dphi = torch.angle(S[0] * torch.conj(S[1]))
    x = torch.stack([torch.cos(dphi), torch.sin(dphi)]) * IPD_SCALE        # (2, 257, T) [cos, sin], L-relative
    return F.interpolate(x.unsqueeze(0), size=(H, W), mode="nearest")[0]   # same resize as _stft_mag


def _ipd1(scene, step):
    return _stft_ipd1(_load_wave1(f"{ROOT}/{scene}/audio_wav/audio_{_stem(step)}.wav"))


_IPD_SIGN = {0: torch.tensor([1.0, 1.0]).view(2, 1, 1), 1: torch.tensor([1.0, -1.0]).view(2, 1, 1)}


def _ipd(scene, front, mode):
    """(N, 2, H, W): per channel of _CH[mode] the planes [cos, sgn_ear * sin] of the pair at that slot."""
    chans = _CH[mode]
    offs = sorted({o for o, _ in chans})
    step = dict(zip(offs, _group_steps(front, offs)))
    ip = {o: _ipd1(scene, step[o]) for o in offs}
    return torch.stack([ip[o] * _IPD_SIGN[e] for o, e in chans], 0)


def _load_depth(scene, front):
    # radial ERP depth (matches OAA's radial output; erp_depth/ is not populated for all scenes)
    d = np.load(f"{ROOT}/{scene}/erp_depth_radial/erp_depth_{_stem(front)}.npy").astype(np.float32)
    t = torch.from_numpy(d)
    if t.dim() == 2:
        t = t[None, None]
    t = F.interpolate(t, size=(H, W), mode="nearest")[0]
    valid = torch.isfinite(t) & (t > 0)
    t = torch.where(valid, t, torch.zeros_like(t))
    return (t / MAX_DEPTH).clamp(0, 1), valid.float()


class _Base(Dataset):
    def __init__(self, split, mode="r2"):
        assert mode in _CH, mode
        self.mode, self.samples = mode, _index(split)

    def __len__(self):
        return len(self.samples)


class RotSet(_Base):
    def __getitem__(self, i):
        sc, f = self.samples[i]
        depth, mask = _load_depth(sc, f)
        d = {"spec": _spec(sc, f, self.mode), "depth": depth, "mask": mask, "idx": i}
        if EMIT_IPD:                      # --pair-feat ipd only; default False -> the dict above, unchanged
            d["ipd"] = _ipd(sc, f, self.mode)
        return d


class WaveSet(_Base):
    def __getitem__(self, i):
        sc, f = self.samples[i]
        depth, mask = _load_depth(sc, f)
        return {"wave": _load_wave(sc, f, self.mode), "depth": depth, "mask": mask}


class SpecWaveSet(_Base):
    """spec is ALWAYS built from the standard WINDOW cut (spec recipe untouched); wave_window
    only lengthens the raw waveform handed to the model's wave branch (EchoDiffusion CIDE)."""

    def __init__(self, split, mode="r2", wave_window=None):
        super().__init__(split, mode)
        self.wave_window = wave_window or WINDOW

    def __getitem__(self, i):
        sc, f = self.samples[i]
        wave_std = _load_wave(sc, f, self.mode)
        wave = wave_std if self.wave_window == WINDOW else _load_wave(sc, f, self.mode, self.wave_window)
        depth, mask = _load_depth(sc, f)
        return {"spec": _stft_mag(wave_std), "wave": wave, "depth": depth, "mask": mask, "idx": i}


def loader(split, batch_size, shuffle, num_workers, mode="r2", *_a, **_k):
    # only the two determinism kwargs are forwarded (train_oaa.py --deterministic); anything else
    # is still swallowed, so every existing caller builds the exact same DataLoader as before
    return DataLoader(RotSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True,
                      **{k: v for k, v in _k.items() if k in ("worker_init_fn", "generator")})


def wave_loader(split, batch_size, shuffle, num_workers, mode="r2"):
    return DataLoader(WaveSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)


def spec_wave_loader(split, batch_size, shuffle, num_workers, mode="r2", wave_window=None):
    return DataLoader(SpecWaveSet(split, mode, wave_window), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)
