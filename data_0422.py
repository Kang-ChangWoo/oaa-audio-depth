"""Replica (replica_0422_lite) data module — on-the-fly STFT, scene-disjoint splits.

Drop-in replacement for data.py (same `loader` / `wave_loader` / `IN_CH` interface) selected at
runtime via `DATA_MODULE=data_0422`. MP3D's data.py is left untouched (both coexist).

Dataset: /root/local1/changwoo/replica_0422_lite/<scene>/{audio_wav,erp_depth}/*  (18 scenes,
400 poses each: audio_NNN.wav stereo 48k, erp_depth_NNN.npy (512,1024) float32 metres).

Split policy (env `R0422_SPLIT`, default "off3" = MP3D-style scene-disjoint, one scene/family
held out; location-level splits leak train<->val neighbours so are NOT used):
  train 12 / val {apartment_1, frl_apartment_4, office_3} / test {apartment_2, frl_apartment_5, office_4}

Spec recipe (matches MP3D's data.py so the model input format is identical across datasets):
  wav[:WINDOW] per channel, torch.stft(n_fft=512, win=400, hop=160, |.|=magnitude) -> (257,T),
  bilinear-resized to (256,512). WINDOW = round-trip window for MAX_DEPTH (279.9 samples/m @48k;
  =2823 at 10 m, matching MP3D). Depth (512,1024) is nearest-resized to (256,512); mask = depth>0 & finite.
"""
import os, json, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = "/root/local1/changwoo/replica_0422_lite"
SR = 48000
MAX_DEPTH = 10.0                                  # same as MP3D (data.py); depth >10 m clamped to 1.0
WINDOW = int(round(2 * MAX_DEPTH / 343.0 * SR))   # ~2799 samples (= MP3D's ~2823 @10 m round-trip)
H, W = 256, 512
N_FFT, WIN, HOP = 512, 400, 160

MODES = ("r2",)                                   # 2ch only for now (multi-view Replica: TODO)
IN_CH = {"r2": 2}

_SPLITS = {
    "off3": {
        "val": ["apartment_1", "frl_apartment_4", "office_3"],
        "test": ["apartment_2", "frl_apartment_5", "office_4"],
    }
}


def _scenes(split):
    policy = os.environ.get("R0422_SPLIT", "off3")
    spec = _SPLITS[policy]
    allsc = sorted(d for d in os.listdir(ROOT)
                   if os.path.isdir(os.path.join(ROOT, d)) and d != "logs")
    if split == "val":
        return spec["val"]
    if split == "test":
        return spec["test"]
    held = set(spec["val"]) | set(spec["test"])
    return [s for s in allsc if s not in held]      # train = remainder (12 scenes)


def _index(split):
    """List of (wav_path, depth_path) pairs for the split's scenes."""
    pairs = []
    for sc in _scenes(split):
        wd, dd = f"{ROOT}/{sc}/audio_wav", f"{ROOT}/{sc}/erp_depth"
        for w in sorted(glob.glob(f"{wd}/audio_*.wav")):
            idx = os.path.basename(w)[len("audio_"):-len(".wav")]
            d = f"{dd}/erp_depth_{idx}.npy"
            if os.path.exists(d):
                pairs.append((w, d))
    return pairs


_HANN = torch.hann_window(WIN)


def _stft_mag(wav2):
    """wav2 (2, WINDOW) float32 -> magnitude STFT (2, 256, 512)."""
    s = torch.stft(wav2, n_fft=N_FFT, hop_length=HOP, win_length=WIN, window=_HANN,
                   center=True, return_complex=True).abs()          # (2, 257, T)
    return F.interpolate(s.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]


def _load_wave(wpath):
    import soundfile as sf
    y, _ = sf.read(wpath, frames=WINDOW, dtype="float32", always_2d=True)   # (<=WINDOW, 2)
    if y.shape[0] < WINDOW:
        y = np.pad(y, ((0, WINDOW - y.shape[0]), (0, 0)))
    return torch.from_numpy(y.T.copy())                                     # (2, WINDOW)


def _load_depth(dpath):
    d = np.load(dpath).astype(np.float32)                                   # (512, 1024)
    t = torch.from_numpy(d)[None, None]
    t = F.interpolate(t, size=(H, W), mode="nearest")[0]                    # (1, 256, 512)
    valid = torch.isfinite(t) & (t > 0)
    t = torch.where(valid, t, torch.zeros_like(t))
    return (t / MAX_DEPTH).clamp(0, 1), valid.float()


class _Base(Dataset):
    def __init__(self, split):
        self.pairs = _index(split)

    def __len__(self):
        return len(self.pairs)


class RotSet(_Base):
    """2ch magnitude-STFT spec + depth/mask (mode r2)."""

    def __init__(self, split, mode="r2"):
        assert mode == "r2", f"data_0422 currently supports r2 (2ch) only, got {mode}"
        super().__init__(split)

    def __getitem__(self, i):
        w, d = self.pairs[i]
        depth, mask = _load_depth(d)
        return {"spec": _stft_mag(_load_wave(w)), "depth": depth, "mask": mask}


class WaveSet(_Base):
    """Raw 2ch waveform (2, WINDOW) + depth/mask, for EchoScan / EchoDiffusion."""

    def __getitem__(self, i):
        w, d = self.pairs[i]
        depth, mask = _load_depth(d)
        return {"wave": _load_wave(w), "depth": depth, "mask": mask}


def loader(split, batch_size, shuffle, num_workers, mode="r2"):
    return DataLoader(RotSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)


def wave_loader(split, batch_size, shuffle, num_workers):
    return DataLoader(WaveSet(split), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)


class SpecWaveSet(_Base):
    """Both 2ch spec AND raw waveform + depth/mask, for EchoDiffusion (needs spec + wave)."""

    def __getitem__(self, i):
        w, d = self.pairs[i]
        wave = _load_wave(w)
        depth, mask = _load_depth(d)
        return {"spec": _stft_mag(wave), "wave": wave, "depth": depth, "mask": mask}


def spec_wave_loader(split, batch_size, shuffle, num_workers):
    return DataLoader(SpecWaveSet(split), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)
