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
import os, json, glob, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = os.environ.get("REPLICA_ROOT", "/root/local2/replica_0422_lite")   # env override for machines where data lives elsewhere
SR = 48000
MAX_DEPTH = 10.0                                  # same as MP3D (data.py); depth >10 m clamped to 1.0
WINDOW = int(round(2 * MAX_DEPTH / 343.0 * SR))   # ~2799 samples (= MP3D's ~2823 @10 m round-trip)
H, W = 256, 512
N_FFT, WIN, HOP = 512, 400, 160

# Replica poses come in groups of 4 consecutive steps = same position, yaws 0/90/180/270
# (verified). A sample is indexed by a "front" step; extra views are group-relative yaw slots
# stacked as [L,R] pairs. Relative to the front, poses are always the MP3D _POOL8 encoding, so
# OAA's pose geometry transfers unchanged.
#   r2 = front binaural            [0L,0R]                          2ch
#   fb = front + back              [0L,0R,180L,180R]  (best 4ch)    4ch
#   r6 = front,+90,+270            [0L,0R,90L,90R,270L,270R]        6ch
#   r8 = all four yaws             [0..270 L/R] (= MP3D _POOL8)     8ch
_OFFS = {"r2": (0,), "fb": (0, 2), "fs": (0, 1), "r6": (0, 1, 3), "r8": (0, 1, 2, 3)}  # group-relative yaw slots (each adds L,R)
# Channel-level specs: list of (yaw-slot, ear) with ear 0=L, 1=R. Pair modes derive from _OFFS;
# mixed-ear modes (single ear per extra yaw) are listed explicitly — e.g. cb = MP3D champion cB.
_CH = {m: [(o, e) for o in offs for e in (0, 1)] for m, offs in _OFFS.items()}
_CH["cb"] = [(0, 0), (0, 1), (1, 1), (3, 0)]        # [0L, 0R, 90R, 270L] 4ch
MODES = tuple(_CH)
IN_CH = {m: len(ch) for m, ch in _CH.items()}       # r2:2, fb/fs/cb:4, r6:6, r8:8
# OAA view_poses per mode (yaw rad, ear sign -1=L/+1=R), relative to the front frame:
_HALF, _PI, _3H = math.pi / 2, math.pi, 3 * math.pi / 2
POSES = {m: [(o * _HALF, (-1.0, 1.0)[e]) for o, e in ch] for m, ch in _CH.items()}

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
    """List of (scene, front_step) samples. Each scene has poses in groups of 4 (yaw 0/90/180/270);
    any step can be the 'front', extra views are pulled from its group (see _OFFS)."""
    samples = []
    for sc in _scenes(split):
        n = len(glob.glob(f"{ROOT}/{sc}/audio_wav/audio_*.wav"))
        n -= n % 4                                     # keep whole 4-yaw groups
        samples += [(sc, i) for i in range(n)]
    return samples


def _group_steps(front, offs):
    """Group-relative view steps for a front step (front=slot 0 of its 4-yaw group)."""
    base, k = 4 * (front // 4), front % 4
    return [base + (k + o) % 4 for o in offs]


_HANN = torch.hann_window(WIN)


def _stft_mag(wav2):
    """wav2 (2, WINDOW) float32 -> magnitude STFT (2, 256, 512)."""
    s = torch.stft(wav2, n_fft=N_FFT, hop_length=HOP, win_length=WIN, window=_HANN,
                   center=True, return_complex=True).abs()          # (2, 257, T)
    return F.interpolate(s.unsqueeze(0), size=(H, W), mode="nearest")[0]   # nearest = MP3D cache recipe


def _load_wave1(wpath):
    import soundfile as sf
    y, _ = sf.read(wpath, frames=WINDOW, dtype="float32", always_2d=True)   # (<=WINDOW, 2)
    if y.shape[0] < WINDOW:
        y = np.pad(y, ((0, WINDOW - y.shape[0]), (0, 0)))
    return torch.from_numpy(y.T.copy())                                     # (2, WINDOW)


def _load_wave(scene, front, mode):
    """Stacked waveform (len(_CH[mode]), WINDOW), channels in _CH[mode] order."""
    chans = _CH[mode]
    offs = sorted({o for o, _ in chans})
    step = dict(zip(offs, _group_steps(front, offs)))
    wav = {o: _load_wave1(f"{ROOT}/{scene}/audio_wav/audio_{step[o]:03d}.wav") for o in offs}
    return torch.cat([wav[o][e:e + 1] for o, e in chans], 0)


def _load_depth(dpath):
    d = np.load(dpath).astype(np.float32)                                   # (512, 1024)
    t = torch.from_numpy(d)[None, None]
    t = F.interpolate(t, size=(H, W), mode="nearest")[0]                    # (1, 256, 512)
    valid = torch.isfinite(t) & (t > 0)
    t = torch.where(valid, t, torch.zeros_like(t))
    return (t / MAX_DEPTH).clamp(0, 1), valid.float()


class _Base(Dataset):
    def __init__(self, split, mode="r2"):
        assert mode in _CH, f"unknown mode {mode}; have {tuple(_CH)}"
        self.mode = mode
        self.samples = _index(split)

    def __len__(self):
        return len(self.samples)

    def _depth(self, scene, front):
        return _load_depth(f"{ROOT}/{scene}/erp_depth/erp_depth_{front:03d}.npy")


class RotSet(_Base):
    """Multi-view magnitude-STFT spec (2*len(views) ch) + depth/mask."""

    def __getitem__(self, i):
        sc, front = self.samples[i]
        depth, mask = self._depth(sc, front)
        return {"spec": _stft_mag(_load_wave(sc, front, self.mode)), "depth": depth, "mask": mask}


class WaveSet(_Base):
    """Multi-view raw waveform (2*len(views), WINDOW) + depth/mask, for EchoScan / EchoDiffusion."""

    def __getitem__(self, i):
        sc, front = self.samples[i]
        depth, mask = self._depth(sc, front)
        return {"wave": _load_wave(sc, front, self.mode), "depth": depth, "mask": mask}


class SpecWaveSet(_Base):
    """Both spec AND raw waveform + depth/mask, for EchoDiffusion (needs spec + wave)."""

    def __getitem__(self, i):
        sc, front = self.samples[i]
        wave = _load_wave(sc, front, self.mode)
        depth, mask = self._depth(sc, front)
        return {"spec": _stft_mag(wave), "wave": wave, "depth": depth, "mask": mask}


def loader(split, batch_size, shuffle, num_workers, mode="r2", *_ignore, **_kw):
    return DataLoader(RotSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)


def wave_loader(split, batch_size, shuffle, num_workers, mode="r2"):
    return DataLoader(WaveSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)


def spec_wave_loader(split, batch_size, shuffle, num_workers, mode="r2"):
    return DataLoader(SpecWaveSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)
