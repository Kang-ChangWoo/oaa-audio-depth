"""MP3D per-step spectrogram cache builder.

Saves per-step binaural magnitude STFT (2,256,512) as float32 npy — all channel modes (r2/fb/fs/r6/r8/cb)
assemble from the same cache. Kept in fp32, so bit-identical to on-the-fly (zero effect on metrics).
Resumable (existing files are skipped). Run:  python tools/build_spec_cache_mp3d.py [--out DIR]
"""
# --- repo-root bootstrap: importable root modules (eval, data_*, model) + relative comparison/ paths
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
_os.chdir(ROOT)
import os, sys, glob, argparse
import numpy as np
from multiprocessing import Pool

import data_mp3d as dm


def do_scene(args):
    sc, out = args
    os.makedirs(f"{out}/{sc}", exist_ok=True)
    n_done = n_new = 0
    for w in sorted(glob.glob(f"{dm.ROOT}/{sc}/audio_wav/audio_*.wav")):
        step = int(os.path.basename(w)[6:-4])
        p = f"{out}/{sc}/spec_{step:03d}.npy"
        if os.path.exists(p):
            n_done += 1
            continue
        spec = dm._stft_mag(dm._load_wave1(w)).numpy().astype(np.float32)   # (2,256,512)
        np.save(p + ".tmp.npy", spec)
        os.replace(p + ".tmp.npy", p)                                       # atomic (avoids partial writes)
        n_new += 1
    return sc, n_done, n_new


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("MP3D_SPEC_CACHE", "cache/mp3d_spec"))
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    scenes = sorted(d for d in os.listdir(dm.ROOT) if os.path.isdir(f"{dm.ROOT}/{d}/audio_wav"))
    print(f"{len(scenes)} scenes -> {a.out}", flush=True)
    with Pool(a.procs) as pool:
        for i, (sc, nd, nn) in enumerate(pool.imap_unordered(do_scene, [(s, a.out) for s in scenes])):
            print(f"[{i+1}/{len(scenes)}] {sc}: skip {nd}, new {nn}", flush=True)
    print("cache build complete", flush=True)
