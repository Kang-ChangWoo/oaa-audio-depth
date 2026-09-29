#!/usr/bin/env python3
"""E142-fix step 1: build the FIXED reference cloud, one per (scene, seq).

Why this exists
---------------
`e114_eval.py` builds its reference cloud from the *selected* steps (the ones present in the
prediction npz), and measures every range to the *selected* camera positions.  So when the budget-8
slice compares r8 N=2 against r2 N=8, the two arms are scored against different targets
(REPORT_E142 sec 6-2a measured 22,353 vs 43,336 voxels = 1.94x) and their distance bins are cut at
different places.  That confound is fatal for the `>4 m` claim.

This script writes, for each (scene, seq), a reference that is a function of (scene, seq) ONLY:
  * `ref_vox`   : GT cloud of EVERY step of the sequence, voxelised at VOXEL/2 (same as the original)
  * `ref_range` : min distance to EVERY camera position of the sequence
  * `cam_all`   : every camera position, so the evaluator can bin predictions on the same partition

Because it depends on neither `mode` nor `N`, every cell of the grid loads the *same file*, so
byte-identity of the scoring target is guaranteed by construction (and proved by md5 per cell).

Nothing here modifies `e114_eval.py`; the geometry helpers are imported from the repo unchanged.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path("/root/storage/implementation/shared_audio/EchoRecon")
sys.path.insert(0, str(REPO / "src"))
from data import TEST_SCENES, Sequence, sequences
from eval_fusion import resize_nearest
from erp import ray_dirs, to_radial
from fuse import voxel_downsample

# --- identical constants to e114_eval.py (do not diverge) ---
VOXEL = 0.1
STRIDE = 2
MAX_DEPTH = 10.0
BIN_EDGES = (0.5, 1.5, 4.0)
BIN_NAMES = ("<0.5", "0.5-1.5", "1.5-4", ">4")
PRED_H, PRED_W = 256, 512  # both posterior heads emit (T, 256, 512); verified for all 39 test seqs


def unproject_radial(depth_radial, pose, dirs, stride=STRIDE, max_depth=MAX_DEPTH):
    """Byte-for-byte the same computation as e114_eval.unproject_radial."""
    from erp import quat_to_R
    d = depth_radial[::stride, ::stride]
    dd = dirs[::stride, ::stride]
    valid = np.isfinite(d) & (d > 0) & (d < max_depth)
    P = dd[valid] * d[valid][:, None]
    R = quat_to_R(pose["rotation"])
    pts = P @ R.T + np.asarray(pose["position"], float)[None, :]
    return pts, d[valid]


def min_range(pts, cams):
    """min_j ||pts - cams[j]||, computed one camera at a time.

    Same values as e114_eval's `norm(pts[None]-cams[:,None], axis=2).min(0)` but O(len(pts)) RAM,
    which matters here because the fixed reference spans up to 60 steps.
    """
    out = None
    for c in cams:
        d = np.linalg.norm(pts - c[None, :], axis=1)
        out = d if out is None else np.minimum(out, d)
    return out


def build(sc, sq, out_root):
    S = Sequence(sc, sq)
    dirs = ray_dirs(PRED_H, PRED_W, "right0")
    pts_list, cams = [], []
    for i in S.steps:
        g = resize_nearest(S.gt_depth(i, "face"), (PRED_H, PRED_W))
        gr = to_radial(g, dirs, "face")
        pts, _ = unproject_radial(gr, S.pose(i), dirs)
        pts_list.append(pts)
        cams.append(np.asarray(S.pose(i)["position"], float))
    ref_pts_raw = np.concatenate(pts_list)
    ref_vox, _ = voxel_downsample(ref_pts_raw, VOXEL / 2)
    cam_all = np.stack(cams)
    ref_range = min_range(ref_vox, cam_all)

    d = out_root / sc
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{sq}.npz"
    # fixed key order + no compression timestamp -> reproducible bytes
    with open(f, "wb") as fh:
        np.savez(fh, ref_vox=ref_vox, ref_range=ref_range, cam_all=cam_all,
                 steps=np.asarray(S.steps))
    md5 = hashlib.md5(f.read_bytes()).hexdigest()

    rb = np.digitize(ref_range, BIN_EDGES)
    per_bin = {n: int((rb == b).sum()) for b, n in enumerate(BIN_NAMES)}
    return {"n_steps": len(S.steps), "n_ref_vox": int(len(ref_vox)),
            "n_ref_raw": int(len(ref_pts_raw)), "md5": md5, "per_bin": per_bin,
            "extent_m": float(np.linalg.norm(cam_all.max(0) - cam_all.min(0)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    out_root = Path(a.out_root)
    man = {"voxel_ref": VOXEL / 2, "stride": STRIDE, "max_depth": MAX_DEPTH,
           "pred_hw": [PRED_H, PRED_W], "bin_edges": list(BIN_EDGES), "per_seq": {}}
    t0 = time.time()
    for sc in TEST_SCENES:
        for sq in sequences(sc):
            info = build(sc, sq, out_root)
            man["per_seq"][f"{sc}/{sq}"] = info
            print(f"{sc}/{sq} steps={info['n_steps']} ref_vox={info['n_ref_vox']} "
                  f"md5={info['md5'][:8]} ({time.time()-t0:.0f}s)", flush=True)
    man["n_sequences"] = len(man["per_seq"])
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest).write_text(json.dumps(man, indent=1))
    print(f"wrote {a.manifest}  ({man['n_sequences']} sequences, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
