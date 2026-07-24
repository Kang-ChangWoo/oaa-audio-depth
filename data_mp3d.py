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
WINDOW = 2823   # MP3D legacy convention (340 m/s round-trip @10 m) — user 2026-07-24: MP3D와 Replica는
#               # 데이터셋별 고유 창 유지 (Replica는 343 m/s = 2799). 캐시 체크포인트와 창 일치.
H, W = 256, 512
N_FFT, WIN, HOP = 512, 400, 160

_OFFS = {"r2": (0,), "fb": (0, 2), "fs": (0, 1), "r6": (0, 1, 3), "r8": (0, 1, 2, 3)}  # group-relative yaw slots (L,R each)
# Channel-level specs (mirror of data_0422): (yaw-slot, ear) with ear 0=L, 1=R. cb = the MP3D
# champion config [0L, 0R, 90R, 270L] so cache-trained cB checkpoints evaluate natively.
_CH = {m: [(o, e) for o in offs for e in (0, 1)] for m, offs in _OFFS.items()}
_CH["cb"] = [(0, 0), (0, 1), (1, 1), (3, 0)]
_CH["cB"] = _CH["cb"]        # eval.py _NV2MODE fallback uses the cache module's capitalisation
MODES = tuple(_CH)
IN_CH = {m: len(ch) for m, ch in _CH.items()}
POSES = {m: [(o * (math.pi / 2), (-1.0, 1.0)[e]) for o, e in ch] for m, ch in _CH.items()}


_NSCENE = {}


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
    wav = {o: _load_wave1(f"{ROOT}/{scene}/audio_wav/audio_{step[o]:03d}.wav", frames) for o in offs}
    return torch.cat([wav[o][e:e + 1] for o, e in chans], 0)


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
        assert mode in _CH, mode
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
        return {"spec": _stft_mag(wave_std), "wave": wave, "depth": depth, "mask": mask}


def loader(split, batch_size, shuffle, num_workers, mode="r2", *_a, **_k):
    return DataLoader(RotSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)


def wave_loader(split, batch_size, shuffle, num_workers, mode="r2"):
    return DataLoader(WaveSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)


def spec_wave_loader(split, batch_size, shuffle, num_workers, mode="r2", wave_window=None):
    return DataLoader(SpecWaveSet(split, mode, wave_window), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)
