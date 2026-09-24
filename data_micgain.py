"""supple_mic_gain data module — Replica rotated binaural set for the microphone-scaling study.

Selected at runtime via DATA_MODULE=data_micgain. Same loader / wave_loader /
spec_wave_loader / IN_CH / POSES interface as data_0422, so the trainer, model and eval
pipeline are untouched.

WHY A SEPARATE MODULE. The released replica_0422 set stores four yaws per position in consecutive
files (audio_NNN.wav, groups of four); this set stores twelve yaws per position under explicit
names (audio_<pos>_<yaw>.wav). The dataset README also records that the renderer has no RNG seed,
so yaw 0 here is NOT bit-identical to replica_0422's rot-0 wav (waveform correlation 0.82-0.98 away
from position 0). The scaling comparison therefore has to be closed INSIDE this set: a k=4 baseline
borrowed from replica_0422 would mix the microphone-count effect with renderer noise.

GEOMETRY. 12 binaural headings on a 30 deg grid (0, 30, ... 330), each contributing an (L, R) pair,
so the channel count is 2 x headings and N_max = 24 channels. Headings are added in the bisection
order recorded in mic_order.json -- 0, 180, 90, 270, 30, 210, 120, 300, 60, 240, 150, 330 -- where
each new yaw falls in the largest remaining gap. Taking the first k entries gives the k-heading set,
so the subsets are NESTED by construction (M2 subset M4 subset ... subset M24) and k=4 reproduces
exactly the released r8 input {0, 90, 180, 270}.

Modes are named m<channels>: m2, m4, ... m24. Mode m8 is the r8 geometry.

Split: off3, identical scenes to data_0422 (train 12 / val 3 / test 3), scene-disjoint.
Spec recipe, STFT env overrides and depth handling are copied from data_0422 verbatim so the input
format matches the rest of the campaign.
"""
import os, json, glob, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = os.environ.get("MICGAIN_ROOT", "/root/storage/supple_mic_gain")
SR = 48000
MAX_DEPTH = 10.0
WINDOW = int(os.environ.get("STFT_WINDOW", 2799))
H, W = 256, 512
N_FFT = int(os.environ.get("STFT_NFFT", 512))
WIN   = int(os.environ.get("STFT_WIN", 400))
HOP   = int(os.environ.get("STFT_HOP", 160))

_ORDER = json.load(open(os.path.join(ROOT, "mic_order.json")))["progressive_order"] \
    if os.path.exists(os.path.join(ROOT, "mic_order.json")) \
    else [0, 180, 90, 270, 30, 210, 120, 300, 60, 240, 150, 330]
N_HEAD = len(_ORDER)                                   # 12 headings -> 24 channels

# mode -> list of (yaw_deg, ear) with ear 0=L, 1=R, in the dataset's own addition order
_CH = {f"m{2*k}": [(y, e) for y in _ORDER[:k] for e in (0, 1)] for k in range(1, N_HEAD + 1)}
MODES = tuple(_CH)
IN_CH = {m: len(ch) for m, ch in _CH.items()}
# OAA view_poses: yaw in radians, ear sign -1=L / +1=R. Yaw is relative to the reference heading,
# which is the geometry OAA's ray-mic attention expects.
POSES = {m: [(math.radians(y), (-1.0, 1.0)[e]) for y, e in ch] for m, ch in _CH.items()}

_SPLITS = {"off3": {"val": ["apartment_1", "frl_apartment_4", "office_3"],
                    "test": ["apartment_2", "frl_apartment_5", "office_4"]}}


def _scenes(split):
    spec = _SPLITS[os.environ.get("R0422_SPLIT", "off3")]
    allsc = sorted(d for d in os.listdir(ROOT)
                   if os.path.isdir(os.path.join(ROOT, d)) and d != "logs")
    if split == "val":
        return spec["val"]
    if split == "test":
        return spec["test"]
    held = set(spec["val"]) | set(spec["test"])
    return [s for s in allsc if s not in held]


def _index(split):
    """(scene, pos, ref_heading) samples. Every rendered yaw serves as a reference heading once, so
    a k-heading sample reads yaw (ref + r) mod 360 for each r in the first k of the addition order.
    Only 30-deg-grid headings are used as references (the extra 45-deg yaws on the test scenes exist
    for a different study and have no 30-deg partners)."""
    out = []
    for sc in _scenes(split):
        pos = sorted({int(os.path.basename(f).split("_")[1])
                      for f in glob.glob(f"{ROOT}/{sc}/audio_wav/audio_*_*.wav")})
        for p in pos:
            for h in _ORDER:                            # every 30-deg heading is a valid reference
                out.append((sc, p, h))
    return out


_HANN = torch.hann_window(WIN)


def _stft_mag(wav2):
    s = torch.stft(wav2, n_fft=N_FFT, hop_length=HOP, win_length=WIN, window=_HANN,
                   center=True, return_complex=True).abs()
    return F.interpolate(s.unsqueeze(0), size=(H, W), mode="nearest")[0]


def _load_wave1(path, frames=None):
    import soundfile as sf
    frames = WINDOW if frames is None else frames
    y, _ = sf.read(path, frames=frames, dtype="float32", always_2d=True)
    if y.shape[0] < frames:
        y = np.pad(y, ((0, frames - y.shape[0]), (0, 0)))
    return torch.from_numpy(y.T.copy())                 # (2, frames)


def _wav(scene, pos, yaw):
    return f"{ROOT}/{scene}/audio_wav/audio_{pos:03d}_{yaw % 360:03d}.wav"


def _load_wave(scene, pos, ref, mode, frames=None):
    """Channels in the mode's order: (yaw, ear) -> the ear channel of file (ref + yaw) mod 360."""
    chans = _CH[mode]
    yaws = sorted({y for y, _ in chans})
    wav = {y: _load_wave1(_wav(scene, pos, ref + y), frames) for y in yaws}
    return torch.cat([wav[y][e:e + 1] for y, e in chans], 0)


def _spec(scene, pos, ref, mode):
    chans = _CH[mode]
    yaws = sorted({y for y, _ in chans})
    sp = {y: _stft_mag(_load_wave1(_wav(scene, pos, ref + y))) for y in yaws}
    return torch.cat([sp[y][e:e + 1] for y, e in chans], 0)


def _load_depth(scene, pos, ref):
    d = np.load(f"{ROOT}/{scene}/erp_depth_radial/erp_depth_{pos:03d}_{ref % 360:03d}.npy").astype(np.float32)
    t = torch.from_numpy(d)
    if t.dim() == 2:
        t = t[None, None]
    t = F.interpolate(t, size=(H, W), mode="nearest")[0]
    valid = torch.isfinite(t) & (t > 0)
    t = torch.where(valid, t, torch.zeros_like(t))
    return (t / MAX_DEPTH).clamp(0, 1), valid.float()


class _Base(Dataset):
    def __init__(self, split, mode="m8"):
        assert mode in _CH, f"unknown mode {mode}; have {tuple(_CH)}"
        self.mode, self.samples = mode, _index(split)

    def __len__(self):
        return len(self.samples)


class RotSet(_Base):
    def __getitem__(self, i):
        sc, pos, ref = self.samples[i]
        depth, mask = _load_depth(sc, pos, ref)
        return {"spec": _spec(sc, pos, ref, self.mode), "depth": depth, "mask": mask, "idx": i}


class WaveSet(_Base):
    def __getitem__(self, i):
        sc, pos, ref = self.samples[i]
        depth, mask = _load_depth(sc, pos, ref)
        return {"wave": _load_wave(sc, pos, ref, self.mode), "depth": depth, "mask": mask}


class SpecWaveSet(_Base):
    """Both spec AND raw waveform + depth/mask, for EchoDiffusion (needs spec + wave).

    Mirrors data_0422.SpecWaveSet. The spec is built from the SAME stacked wave, which is
    bit-identical to the per-yaw _spec() path (verified 2026-09-24: max|diff| = 0.0) because
    _stft_mag is per-channel independent. Added after the eco-scale queue died 141 times on
    `AttributeError: module 'data_micgain' has no attribute 'spec_wave_loader'`: neither
    loader (no "wave") nor wave_loader (no "spec"/"idx") satisfies train_echodiffusion.py.
    """

    def __getitem__(self, i):
        sc, pos, ref = self.samples[i]
        wave = _load_wave(sc, pos, ref, self.mode)
        depth, mask = _load_depth(sc, pos, ref)
        return {"spec": _stft_mag(wave), "wave": wave, "depth": depth, "mask": mask, "idx": i}


def loader(split, batch_size, shuffle, num_workers, mode="m8", *_a, **_k):
    return DataLoader(RotSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)


def wave_loader(split, batch_size, shuffle, num_workers, mode="m8", *_a, **_k):
    return DataLoader(WaveSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)


def spec_wave_loader(split, batch_size, shuffle, num_workers, mode="m8", *_a, **_k):
    return DataLoader(SpecWaveSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)
