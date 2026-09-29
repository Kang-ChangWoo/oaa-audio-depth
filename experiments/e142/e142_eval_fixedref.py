#!/usr/bin/env python3
"""E142-fix step 2: `e114_eval.py` with the reference cloud decoupled from the budget.

This is a NEW FILE.  `e114_eval.py` is not modified, not imported and not shadowed; its md5 is
re-checked by the runner before and after this grid.

Exactly two things differ from `e114_eval.py`, and nothing else:

  (1) the reference cloud is loaded from the pre-built per-(scene, seq) cache instead of being
      rebuilt from the selected steps, so it is a function of (scene, seq) ONLY -- identical for
      r2 and r8, and identical for every N;
  (2) the ranges used for distance binning (both on the reference side and on the prediction side)
      are measured to EVERY camera position of the sequence instead of only the selected ones, so
      the four distance bins are a fixed partition of space per (scene, seq).

(2) is not optional: if the reference were binned on the fixed camera set while predictions were
binned on the selected set, a reference voxel and a predicted voxel at the same location could fall
in different bins and the per-bin precision/recall would be computed over inconsistent sets.

TAU / VOXEL / STRIDE / MAX_DEPTH / FRAC / BIN_EDGES and the fusion + top-FRAC rule are copied from
`e114_eval.py` verbatim.  Self-consistency gate: for the N=all cells the selected set IS the full
set, so this script must reproduce the `e114_eval.py` N=all numbers exactly.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

REPO = Path("/root/storage/implementation/shared_audio/EchoRecon")
sys.path.insert(0, str(REPO / "src"))
from data import TEST_SCENES, Sequence, sequences
from erp import quat_to_R, ray_dirs, to_radial
from fuse import voxel_downsample

# --- verbatim from e114_eval.py ---
TAU = 0.2
VOXEL = 0.1
STRIDE = 2
MAX_DEPTH = 10.0
FRAC = 0.25
BIN_EDGES = (0.5, 1.5, 4.0)
BIN_NAMES = ("<0.5", "0.5-1.5", "1.5-4", ">4")


def unproject_radial(depth_radial, pose, dirs, stride=STRIDE, max_depth=MAX_DEPTH):
    d = depth_radial[::stride, ::stride]
    dd = dirs[::stride, ::stride]
    valid = np.isfinite(d) & (d > 0) & (d < max_depth)
    P = dd[valid] * d[valid][:, None]
    R = quat_to_R(pose["rotation"])
    pts = P @ R.T + np.asarray(pose["position"], float)[None, :]
    return pts, d[valid]


def bin_metrics(pred_pts, pred_range, ref_pts, ref_range, tau=TAU):
    if len(pred_pts) == 0 or len(ref_pts) == 0:
        return None
    d_pg, _ = cKDTree(ref_pts).query(pred_pts, k=1, workers=-1)
    d_gp, _ = cKDTree(pred_pts).query(ref_pts, k=1, workers=-1)
    p = float((d_pg < tau).mean()); r = float((d_gp < tau).mean())
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    out = {"overall": dict(precision=p, recall=r, f1=f1, n_pred=int(len(pred_pts)), n_ref=int(len(ref_pts)))}
    pb = np.digitize(pred_range, BIN_EDGES); rb = np.digitize(ref_range, BIN_EDGES)
    for b, name in enumerate(BIN_NAMES):
        pm = pb == b; rm = rb == b
        npred = int(pm.sum()); nref = int(rm.sum())
        if npred == 0 or nref == 0:
            out[name] = dict(precision=float("nan"), recall=float("nan"), f1=float("nan"), n_pred=npred, n_ref=nref)
            continue
        p_b = float((d_pg[pm] < tau).mean()); r_b = float((d_gp[rm] < tau).mean())
        f1_b = 2 * p_b * r_b / (p_b + r_b) if p_b + r_b > 0 else 0.0
        out[name] = dict(precision=p_b, recall=r_b, f1=f1_b, n_pred=npred, n_ref=nref)
    return out


def min_range(pts, cams):
    """Same values as norm(pts[None]-cams[:,None], axis=2).min(0), one camera at a time."""
    out = None
    for c in cams:
        d = np.linalg.norm(pts - c[None, :], axis=1)
        out = d if out is None else np.minimum(out, d)
    return out


def kept_and_range(depth_field_per_step, steps, S, dirs, cam_for_range, frac):
    """Prediction side: fusion + top-FRAC rule verbatim from e114_eval.py.

    Only the camera set used for ranging is different (fixed sequence-wide set, see module docstring).
    """
    pts_list = []
    for k, i in enumerate(steps):
        pts, rng = unproject_radial(depth_field_per_step[k], S.pose(i), dirs, STRIDE, MAX_DEPTH)
        pts_list.append(pts)
    pts_all = np.concatenate(pts_list)
    vox, cnt = voxel_downsample(pts_all, VOXEL)
    order = np.argsort(-cnt)
    keep = order[: max(1, int(frac * len(vox)))]
    kept = vox[keep]
    kept_range = min_range(kept, cam_for_range)
    return kept, kept_range


def process_sequence(sc, sq, pred_dir, ref_root, frac=FRAC):
    pf = Path(pred_dir) / sc / f"{sq}.npz"
    if not pf.exists():
        return None, None
    rf = Path(ref_root) / sc / f"{sq}.npz"
    ref_md5 = hashlib.md5(rf.read_bytes()).hexdigest()
    R = np.load(rf)
    ref_vox, ref_range, cam_all = R["ref_vox"], R["ref_range"], R["cam_all"]

    z = np.load(pf)
    P = z["pred"].astype(np.float32)
    steps = z["steps"].tolist()
    S = Sequence(sc, sq)
    T, H, W = P.shape
    dirs = ray_dirs(H, W, "right0")

    pred_radial_full = [to_radial(P[k], dirs, "face") for k in range(len(steps))]
    kept, kept_range = kept_and_range(pred_radial_full, steps, S, dirs, cam_all, frac)
    m = bin_metrics(kept, kept_range, ref_vox, ref_range)
    if m is not None:
        m["_ref"] = {"md5": ref_md5, "n_ref_vox": int(len(ref_vox)),
                     "n_cam_all": int(len(cam_all)), "n_steps_used": int(len(steps))}
    return m, ref_md5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--ref-root", required=True)
    ap.add_argument("--out", required=True)
    # registered primary is FRAC = 0.25; the flag exists only for the labelled sensitivity check
    # of how much the fixed-reference result depends on the top-FRAC rule (REPORT sec 11).
    ap.add_argument("--frac", type=float, default=FRAC)
    a = ap.parse_args()

    per_seq = {}
    ref_md5s = {}
    t0 = time.time()
    for sc in TEST_SCENES:
        for sq in sequences(sc):
            m, md5 = process_sequence(sc, sq, a.pred_dir, a.ref_root, a.frac)
            if m is None:
                print(f"[skip] {sc}/{sq} missing pred", flush=True)
                continue
            per_seq[f"{sc}/{sq}"] = m
            ref_md5s[f"{sc}/{sq}"] = md5
            print(f"{sc}/{sq} f1={m['overall']['f1']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    vals = {"overall": [], **{n: [] for n in BIN_NAMES}}
    for seq, m in per_seq.items():
        for k in vals:
            if m[k] is not None and not np.isnan(m[k]["f1"]):
                vals[k].append(m[k])
    agg = {}
    for k, lst in vals.items():
        if not lst:
            agg[k] = None; continue
        agg[k] = {stat: float(np.mean([x[stat] for x in lst])) for stat in ("precision", "recall", "f1")}
        agg[k]["n_seq"] = len(lst)

    out = {"pred_dir": str(a.pred_dir), "ref_root": str(a.ref_root), "protocol": "fixedref",
           "frac": a.frac, "n_sequences": len(per_seq), "ref_md5": ref_md5s,
           "agg": agg, "per_seq": per_seq}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(agg, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
