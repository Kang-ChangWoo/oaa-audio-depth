"""Matterport3D data module — same on-the-fly pipeline as data_0422, on the ORIGINAL MP3D split.

Selected at runtime via DATA_MODULE=data_mp3d. Reuses the exact modes / STFT recipe / pose
convention as data_0422 (so models transfer unchanged); only the data root and the split differ.

Dataset: /root/local1/changwoo/matterport3d_0303renew/<scene>/{audio_wav,erp_depth}/*  (90 scenes,
400 poses each: audio_NNN.wav stereo 48k, erp_depth_NNN.npy metres; poses in groups of 4 yaws).
Split: the ORIGINAL MP3D train/val/test used by this project, read from the existing sample keys
(cache/rx_wave/{split}_keys.json, "scene/idx" -> 28800 / 3543 / 3600, scene-disjoint).
"""
import os, json, glob, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = "/root/local1/changwoo/matterport3d_0303renew"
KEYS = "/root/storage/implementation/shared_audio/test_for_audio_tof/cache/rx_wave"   # {split}_keys.json
SR = 48000
MAX_DEPTH = 10.0
WINDOW = int(round(2 * MAX_DEPTH / 343.0 * SR))   # ~2799 samples (10 m round-trip; matches data_0422/MP3D)
H, W = 256, 512
N_FFT, WIN, HOP = 512, 400, 160

_OFFS = {"r2": (0,), "fb": (0, 2), "r6": (0, 1, 3), "r8": (0, 1, 2, 3)}   # group-relative yaw slots (L,R each)
MODES = tuple(_OFFS)
IN_CH = {m: 2 * len(o) for m, o in _OFFS.items()}
POSES = {m: [(o * (math.pi / 2), e) for o in offs for e in (-1.0, 1.0)] for m, offs in _OFFS.items()}


def _index(split):
    """Original MP3D split samples as (scene, front_step) from the cached key list."""
    keys = json.load(open(f"{KEYS}/{split}_keys.json"))
    out = []
    for k in keys:
        sc, idx = k.split("/")
        out.append((sc, int(idx)))
    return out


def _group_steps(front, offs):
    base, k = 4 * (front // 4), front % 4
    return [base + (k + o) % 4 for o in offs]


_HANN = torch.hann_window(WIN)


def _stft_mag(wav2):
    s = torch.stft(wav2, n_fft=N_FFT, hop_length=HOP, win_length=WIN, window=_HANN,
                   center=True, return_complex=True).abs()
    return F.interpolate(s.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]


def _load_wave1(wpath):
    import soundfile as sf
    y, _ = sf.read(wpath, frames=WINDOW, dtype="float32", always_2d=True)
    if y.shape[0] < WINDOW:
        y = np.pad(y, ((0, WINDOW - y.shape[0]), (0, 0)))
    return torch.from_numpy(y.T.copy())


def _load_wave(scene, front, mode):
    steps = _group_steps(front, _OFFS[mode])
    return torch.cat([_load_wave1(f"{ROOT}/{scene}/audio_wav/audio_{s:03d}.wav") for s in steps], 0)


def _load_depth(scene, front):
    # radial ERP depth (matches OAA's radial output; erp_depth/ is not populated for all scenes)
    d = np.load(f"{ROOT}/{scene}/erp_depth_radial/erp_depth_{front:03d}.npy").astype(np.float32)
    t = torch.from_numpy(d)
    if t.dim() == 2:
        t = t[None, None]
    t = F.interpolate(t, size=(H, W), mode="nearest")[0]
    valid = torch.isfinite(t) & (t > 0)
    t = torch.where(valid, t, torch.zeros_like(t))
    return (t / MAX_DEPTH).clamp(0, 1), valid.float()


class _Base(Dataset):
    def __init__(self, split, mode="r2"):
        assert mode in _OFFS, mode
        self.mode, self.samples = mode, _index(split)

    def __len__(self):
        return len(self.samples)


class RotSet(_Base):
    def __getitem__(self, i):
        sc, f = self.samples[i]
        depth, mask = _load_depth(sc, f)
        return {"spec": _stft_mag(_load_wave(sc, f, self.mode)), "depth": depth, "mask": mask}


class WaveSet(_Base):
    def __getitem__(self, i):
        sc, f = self.samples[i]
        depth, mask = _load_depth(sc, f)
        return {"wave": _load_wave(sc, f, self.mode), "depth": depth, "mask": mask}


class SpecWaveSet(_Base):
    def __getitem__(self, i):
        sc, f = self.samples[i]
        wave = _load_wave(sc, f, self.mode)
        depth, mask = _load_depth(sc, f)
        return {"spec": _stft_mag(wave), "wave": wave, "depth": depth, "mask": mask}


def loader(split, batch_size, shuffle, num_workers, mode="r2", *_a, **_k):
    return DataLoader(RotSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)


def wave_loader(split, batch_size, shuffle, num_workers, mode="r2"):
    return DataLoader(WaveSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)


def spec_wave_loader(split, batch_size, shuffle, num_workers, mode="r2"):
    return DataLoader(SpecWaveSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)
