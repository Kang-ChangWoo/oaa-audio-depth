#!/usr/bin/env python3
"""E142 geometry / reference-size diagnostics (PREREG sec 2-5, sec 5-3).

Step selections are mode-independent, so these numbers are shared by the r2 and r8 arms of the
same (N, seed) cell -- computed once, not twice.

  --extent-only : positions only (seconds).  Otherwise also builds the reference cloud the way
                  e114_eval.py does, to report how the scoring target grows with N.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path("/root/storage/implementation/shared_audio/EchoRecon")
sys.path.insert(0, str(REPO / "src"))
from data import Sequence                      # noqa: E402
from eval_fusion import resize_nearest         # noqa: E402
from erp import ray_dirs, to_radial, quat_to_R  # noqa: E402
from fuse import voxel_downsample             # noqa: E402

VOXEL, STRIDE, MAX_DEPTH = 0.1, 2, 10.0


def ref_voxels(S, steps, H=256, W=512):
    dirs = ray_dirs(H, W, "right0")
    pts_all = []
    for i in steps:
        g = resize_nearest(S.gt_depth(int(i), "face"), (H, W))
        gr = to_radial(g, dirs, "face")
        d = gr[::STRIDE, ::STRIDE]
        dd = dirs[::STRIDE, ::STRIDE]
        valid = np.isfinite(d) & (d > 0) & (d < MAX_DEPTH)
        P = dd[valid] * d[valid][:, None]
        R = quat_to_R(S.pose(int(i))["rotation"])
        pts_all.append(P @ R.T + np.asarray(S.pose(int(i))["position"], float)[None, :])
    vox, _ = voxel_downsample(np.concatenate(pts_all), VOXEL / 2)
    return len(vox)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-json", required=True, help="a *.select.json from e142_make_subset.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extent-only", action="store_true")
    a = ap.parse_args()

    sel = json.loads(Path(a.select_json).read_text())
    rows = {}
    for key, info in sel["per_seq"].items():
        sc, sq = key.split("/")
        S = Sequence(sc, sq)
        steps = info["steps"]
        pos = np.array([S.pose(int(i))["position"] for i in steps], float)
        if len(pos) > 1:
            dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
            extent = float(dmat.max())
            nnsp = float(np.sort(dmat + np.eye(len(pos)) * 1e9, axis=1)[:, 0].mean())
        else:
            extent, nnsp = 0.0, 0.0
        row = {"n_used": info["n_used"], "extent_m": extent, "mean_nn_spacing_m": nnsp}
        if not a.extent_only:
            row["n_ref_vox"] = int(ref_voxels(S, steps))
        rows[key] = row
        print(f"{key} n={row['n_used']} extent={extent:.3f}m"
              + ("" if a.extent_only else f" ref_vox={row['n_ref_vox']}"), flush=True)

    agg = {"cell": sel["cell"], "nviews": sel["nviews"], "seed": sel["seed"],
           "n_sequences": len(rows),
           "mean_extent_m": float(np.mean([r["extent_m"] for r in rows.values()])),
           "mean_nn_spacing_m": float(np.mean([r["mean_nn_spacing_m"] for r in rows.values()])),
           "mean_n_used": float(np.mean([r["n_used"] for r in rows.values()]))}
    if not a.extent_only:
        agg["mean_n_ref_vox"] = float(np.mean([r["n_ref_vox"] for r in rows.values()]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"agg": agg, "per_seq": rows}, indent=1))
    print(json.dumps(agg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
