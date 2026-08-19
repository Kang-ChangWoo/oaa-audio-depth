"""Precompute CIDE wav2vec2 embeddings per sample (CIDE caching, 2026-07-25).

Justification: in the original code the channel-mixing conv output only enters a no_grad block, so it gets no
gradient (a dead, randomly-initialised frozen parameter), and wav2vec2 is frozen too -> this path is a fixed
function independent of training. The cache uses a canonical channel-mean mixing instead of conv; the trainable
parts (fc/embeddings/adapter) are left untouched.

  HF_HOME=<hf_cache> DATA_MODULE=data_mp3d CUDA_VISIBLE_DEVICES=1 \
    <echodiff_env>/bin/python tools/build_cide_cache.py --mode r2 --waveset std

Output: cache_cide/{DATA_MODULE}_{mode}_{waveset}_{split}.npy  (N, 768) fp16
  waveset std = mean of the front 2ch / all = mean over all channels of the mode (for the wall variant)
"""
# --- repo-root bootstrap: importable root modules (eval, data_*, model) + relative comparison/ paths
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
_os.chdir(ROOT)
import os, argparse
import numpy as np
import torch
from transformers import Wav2Vec2Model

from core.data import get_data_module
_DM = get_data_module()

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="r2")
ap.add_argument("--waveset", default="std", choices=["std", "all"])
ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
ap.add_argument("--batch", type=int, default=64)
a = ap.parse_args()

dev = torch.device("cuda")
w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(dev).eval()
outdir = os.path.join(ROOT, "cache_cide")
os.makedirs(outdir, exist_ok=True)
dm_name = os.environ.get("DATA_MODULE", "data_mp3d")

MIN_LEN = int(w2v.config.inputs_to_logits_ratio * 10 * 2)   # same minimum-length padding rule as CIDE

for split in a.splits:
    ld = _DM.spec_wave_loader(split, a.batch, False, 6, a.mode)
    out = []
    with torch.no_grad():
        for b in ld:
            wv = b["wave"]                                   # (B, C, W)
            x = (wv[:, :2] if a.waveset == "std" else wv).mean(1).to(dev)   # canonical channel mean (B, W)
            if x.shape[1] < MIN_LEN:
                x = torch.nn.functional.pad(x, (0, MIN_LEN - x.shape[1]))
            e = w2v(x).last_hidden_state.mean(1)             # (B, 768)
            out.append(e.half().cpu())
    arr = torch.cat(out).numpy()
    path = f"{outdir}/{dm_name}_{a.mode}_{a.waveset}_{split}.npy"
    np.save(path, arr)
    print(f"[cide-cache] {path} {arr.shape}", flush=True)
