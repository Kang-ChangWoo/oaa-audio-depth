"""Replica per-step spectrogram cache builder (Replica version of tools/build_spec_cache_mp3d.py).

Per-step binaural magnitude STFT (2,256,512) as float32 npy — same WINDOW=2799 (343 m/s) recipe,
bit-identical to on-the-fly. Resumable. Run:  python tools/build_spec_cache_replica.py
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

import data_0422 as dr


def do_scene(args):
    sc, out = args
    os.makedirs(f"{out}/{sc}", exist_ok=True)
    n_done = n_new = 0
    for w in sorted(glob.glob(f"{dr.ROOT}/{sc}/audio_wav/audio_*.wav")):
        step = int(os.path.basename(w)[6:-4])
        p = f"{out}/{sc}/spec_{step:03d}.npy"
        if os.path.exists(p):
            n_done += 1
            continue
        spec = dr._stft_mag(dr._load_wave1(w)).numpy().astype(np.float32)
        np.save(p + ".tmp.npy", spec)
        os.replace(p + ".tmp.npy", p)
        n_new += 1
    return sc, n_done, n_new


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("REPLICA_SPEC_CACHE", "cache/replica_spec"))
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    scenes = sorted(d for d in os.listdir(dr.ROOT)
                    if os.path.isdir(f"{dr.ROOT}/{d}/audio_wav"))
    print(f"{len(scenes)} scenes -> {a.out}", flush=True)
    with Pool(a.procs) as pool:
        for i, (sc, nd, nn) in enumerate(pool.imap_unordered(do_scene, [(s, a.out) for s in scenes])):
            print(f"[{i+1}/{len(scenes)}] {sc}: skip {nd}, new {nn}", flush=True)
    print("cache build complete", flush=True)
