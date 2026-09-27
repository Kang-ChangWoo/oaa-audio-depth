#!/usr/bin/env python3
"""E146 gate 0: the train-split bin histogram recomputed in RADIAL metres.

gate0_bin_dist.py (E114) globbed `erp_depth/` and binned the stored value directly. That value is
per-face cubemap z-depth (planar), but `e114_eval.py` bins the RADIAL range of voxels
(EchoRecon/src/erp.py `to_radial(d, dirs, "face") = d / face_cos`). So the training-side histogram
and the evaluation-side histogram live in different distance spaces.

This script is E114's script with exactly one change: every depth array is divided by `face_cos`
before binning. `face_cos` is imported from EchoRecon (NOT reimplemented) so the treatment uses the
same geometry the eval uses.

Two grids are reported:
  native  512x1024 -- the pixel population E114's histogram used (single-variable change vs E114)
  train   256x512  -- the grid the trainer actually bins on (data_0422._load_depth nearest-resizes
                      512x1024 -> 256x512, and e114_eval builds `dirs` at the prediction resolution
                      256x512 too). This is the grid the E146 loss weight will use.

The `native` histogram is what lands in gate0_bin_dist_radial.json (single-variable change vs the
E114 file that E143/E144/E145 read); `train` is reported alongside so the report can state how far
apart they are.
"""
import os, glob, json, sys
import numpy as np

sys.path.insert(0, "/root/storage/implementation/shared_audio/EchoRecon/src")
from erp import ray_dirs, face_cos            # reused verbatim, not reimplemented

ROOT = "/root/storage/replica_0422"
VAL = {"apartment_1", "frl_apartment_4", "office_3"}
TEST = {"apartment_2", "frl_apartment_5", "office_4"}
held = VAL | TEST
allsc = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and d != "logs")
train_scenes = [s for s in allsc if s not in held]
print("train scenes:", train_scenes, len(train_scenes), flush=True)

edges = (0.5, 1.5, 4.0)
names = ("<0.5", "0.5-1.5", "1.5-4", ">4")
CONV = "right0"                                # e114_eval.py:82 ray_dirs(H, W, "right0")

FC = {}


def fc(h, w):
    if (h, w) not in FC:
        FC[(h, w)] = face_cos(ray_dirs(h, w, CONV)).astype(np.float32)
    return FC[(h, w)]


def binup(v, counts):
    counts[0] += (v < edges[0]).sum()
    counts[1] += ((v >= edges[0]) & (v < edges[1])).sum()
    counts[2] += ((v >= edges[1]) & (v < edges[2])).sum()
    counts[3] += (v >= edges[2]).sum()


grids = {
    "native_planar": dict(counts=np.zeros(4, np.int64), total=0, sum_sqrt=0.0),
    "native_radial": dict(counts=np.zeros(4, np.int64), total=0, sum_sqrt=0.0),
    "train_planar":  dict(counts=np.zeros(4, np.int64), total=0, sum_sqrt=0.0),
    "train_radial":  dict(counts=np.zeros(4, np.int64), total=0, sum_sqrt=0.0),
}
# how many truly->4m radial pixels the PLANAR binning files as 1.5-4 (the brief's 33.47%)
mis_far_planar_says_mid = 0
n_far_radial = 0
nfiles = 0

for sc in train_scenes:
    files = sorted(glob.glob(f"{ROOT}/{sc}/erp_depth/erp_depth_*.npy"))
    for f in files:
        d = np.load(f).astype(np.float32)                       # (512, 1024) planar
        H, W = d.shape
        # --- native grid (E114's population) ---
        c = fc(H, W)
        valid = np.isfinite(d) & (d > 0)
        vp = np.clip(d[valid], 0.05, 10.0)
        vr = np.clip((d / c)[valid], 0.05, 10.0)
        g = grids["native_planar"]; g["total"] += vp.size; g["sum_sqrt"] += np.sqrt(vp).sum(); binup(vp, g["counts"])
        g = grids["native_radial"]; g["total"] += vr.size; g["sum_sqrt"] += np.sqrt(vr).sum(); binup(vr, g["counts"])
        far_r = vr >= edges[2]
        n_far_radial += int(far_r.sum())
        mis_far_planar_says_mid += int((far_r & (vp >= edges[1]) & (vp < edges[2])).sum())
        # --- trainer grid: data_0422._load_depth F.interpolate(mode="nearest") 512x1024 -> 256x512
        #     picks src index floor(dst * 2) = 2*dst, i.e. d[::2, ::2] ---
        dt = d[::2, ::2]
        ct = fc(H // 2, W // 2)
        vt = np.isfinite(dt) & (dt > 0)
        tp = np.clip(dt[vt], 0.05, 10.0)
        tr = np.clip((dt / ct)[vt], 0.05, 10.0)
        g = grids["train_planar"]; g["total"] += tp.size; g["sum_sqrt"] += np.sqrt(tp).sum(); binup(tp, g["counts"])
        g = grids["train_radial"]; g["total"] += tr.size; g["sum_sqrt"] += np.sqrt(tr).sum(); binup(tr, g["counts"])
        nfiles += 1
    print(f"{sc} done ({nfiles} files)", flush=True)

out = {"bin_edges": list(edges), "bin_names": list(names), "convention": CONV,
       "n_train_scenes": len(train_scenes), "n_files": nfiles,
       "radial_far_pixels_binned_as_mid_by_planar_frac": mis_far_planar_says_mid / max(n_far_radial, 1),
       "grids": {}}

for k, g in grids.items():
    frac = g["counts"] / g["total"]
    inv = 1.0 / frac
    bin_w = inv / inv.dot(frac)
    out["grids"][k] = {"bin_frac_train": frac.tolist(), "invfreq_weight_train": bin_w.tolist(),
                       "mean_sqrt_d_train": float(g["sum_sqrt"] / g["total"]),
                       "total_valid_pixels_train": int(g["total"])}
    print(f"\n=== {k} ===")
    for n, cc, ff, ww in zip(names, g["counts"], frac, bin_w):
        print(f"{n:>10s}: {cc:>13d}  {ff*100:7.4f}%   invfreq_w={ww:.4f}   mass={ff*ww:.4f}")
    print(f"total valid pixels: {g['total']}   mean_sqrt={g['sum_sqrt']/g['total']:.6f}")

print(f"\nradial >4m pixels that PLANAR bins as 1.5-4: "
      f"{mis_far_planar_says_mid}/{n_far_radial} = {100*mis_far_planar_says_mid/max(n_far_radial,1):.2f}%")

# the file the E146 trainer reads: same shape/keys as gate0_bin_dist.json, native grid, radial space
nr = out["grids"]["native_radial"]
main = {"bin_edges": list(edges), "bin_names": list(names),
        "bin_frac_train": nr["bin_frac_train"], "invfreq_weight_train": nr["invfreq_weight_train"],
        "mean_sqrt_d_train": nr["mean_sqrt_d_train"],
        "total_valid_pixels_train": nr["total_valid_pixels_train"],
        "distance_space": "radial", "grid": "native_512x1024", "convention": CONV,
        "derived_from": "gate0_bin_dist.py (E114) with depth /= face_cos(ray_dirs(H,W,'right0'))"}
with open("/root/storage/e146_code/gate0_bin_dist_radial.json", "w") as fp:
    json.dump(main, fp, indent=2)
with open("/root/local1/changwoo/e146/results/gate0_bin_dist_radial_full.json", "w") as fp:
    json.dump(out, fp, indent=2)
print("\nsaved gate0_bin_dist_radial.json + results/gate0_bin_dist_radial_full.json")
