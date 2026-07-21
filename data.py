"""Data loading: precomputed magnitude-STFT caches -> (spec, depth, mask) batches.

Cache format (all .npy, mmap-read per index — no on-the-fly audio processing):
  RAD/{split}_spec.npy       (N, 2, 256, 512) float16   0°-yaw binaural magnitude STFT
  REAL90/{split}_spec_off1   (N, 2, 256, 512)           +90°-yaw real binaural
  REAL90/{split}_spec_off2   (N, 2, 256, 512)           +180°
  REAL90/{split}_spec_off3   (N, 2, 256, 512)           +270°
  RAD/{split}_depth.npy      (N, 1, 256, 512)           ERP radial depth, [0,1] (×10 m)
  RAD/{split}_mask.npy       (N, 1, 256, 512)           valid-pixel mask {0,1}

STFT recipe (how the caches were built): wav[:2823 samples] (=2×10m/340 m/s ×48kHz round-trip
window), torchaudio Spectrogram(n_fft=512, win=400, hop=160, power=1.0) -> (257, 18),
nearest-interpolated to (256, 512). No log, no normalisation, no augmentation.

Loader modes (channel order matches oaa._POOL8 pose order):
  r2 = [0L, 0R]                        2ch
  cB = [0L, 0R, 90R, 270L]             4ch  (historic champion config)
  r6 = [0L, 0R, 90L, 90R, 270L, 270R]  6ch
  r8 = all eight                        8ch  (best: OAA 0.718)
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

RAD = "/root/implicit_full_cache/ic2_256x512_nolog"                                  # 0° spec + depth/mask
REAL90 = "/root/storage/implementation/shared_audio/test_for_audio_tof/cache/real_rot"  # 90/180/270° specs

MODES = ("r2", "cB", "r6", "r8")
IN_CH = {"r2": 2, "cB": 4, "r6": 6, "r8": 8}


class RotSet(Dataset):
    def __init__(self, split, mode):
        assert mode in MODES, mode
        self.mode = mode
        self.x0 = np.load(f"{RAD}/{split}_spec.npy", mmap_mode="r")
        if mode != "r2":
            self.x90 = np.load(f"{REAL90}/{split}_spec_off1.npy", mmap_mode="r")
            self.x270 = np.load(f"{REAL90}/{split}_spec_off3.npy", mmap_mode="r")
        if mode == "r8":
            self.x180 = np.load(f"{REAL90}/{split}_spec_off2.npy", mmap_mode="r")
        self.d = np.load(f"{RAD}/{split}_depth.npy", mmap_mode="r")
        self.m = np.load(f"{RAD}/{split}_mask.npy", mmap_mode="r")

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        t = lambda a, sl=slice(None): torch.from_numpy(np.array(a[i][sl], dtype=np.float32))
        x0 = t(self.x0)
        if self.mode == "r2":
            s = x0
        elif self.mode == "cB":                       # 0L,0R + 90R + 270L
            s = torch.cat([x0, t(self.x90, slice(1, 2)), t(self.x270, slice(0, 1))], 0)
        elif self.mode == "r6":                       # 0LR + 90LR + 270LR
            s = torch.cat([x0, t(self.x90), t(self.x270)], 0)
        else:                                         # r8: 0LR + 90LR + 180LR + 270LR (= _POOL8 order)
            s = torch.cat([x0, t(self.x90), t(self.x180), t(self.x270)], 0)
        return {"spec": s, "depth": t(self.d), "mask": t(self.m)}


def loader(split, batch_size, shuffle, num_workers, mode):
    return DataLoader(RotSet(split, mode), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)
