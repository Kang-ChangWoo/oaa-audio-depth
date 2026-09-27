#!/usr/bin/env python3
"""E146 integrity check I1' -- population-matched version of I1.

I1 as written in PREREG_E146 section 4 compared 320 head-of-index loader samples against a
FULL-train-split reference (4.1542 %). Those are different populations: the same 320 samples give a
PLANAR fraction of 2.3177 % against a full-split planar reference of 2.7820 %, i.e. the subsample is
0.46 pp off on the untreated side too. The gate failed on sampling, not on the treatment.

This version removes the mismatch: it walks the TRAINING SAMPLE INDEX the trainer itself uses
(data_0422._index("train")), loads depth through the trainer's own `_load_depth` (so the 512x1024 ->
256x512 nearest resize is the real one), and computes the E145 and E146 expressions on the same
tensors. No STFT, so a full pass is cheap.

Reference values come from gate0_bin_dist_radial.py's `train_*` grids (256x512, whole train split).
Both the planar and the radial number are checked: the planar one must reproduce E145's bin space
(that is the untreated control for this gate), the radial one must land on the treated value.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, "/root/storage/e146_code")
import train_oaa_e146 as T                       # face_cos_like as shipped
import data_0422 as D                            # the trainer's own depth loader

MAX_DEPTH = 10.0
EDGES = torch.tensor([0.5, 1.5, 4.0])
NAMES = ("<0.5", "0.5-1.5", "1.5-4", ">4")
REF = {"planar": 2.7820, "radial": 4.1542}       # train_planar / train_radial >4 m, whole split
TOL = 0.05                                        # pp -- same population now, so this can be tight

samples = D._index("train")
print(f"[cfg] BIN_DIST_PATH={T.BIN_DIST_PATH} ERP_CONV={T.ERP_CONV} ROOT={D.ROOT}", flush=True)
print(f"[data] train index: {len(samples)} samples", flush=True)

cnt_p = torch.zeros(4, dtype=torch.float64)
cnt_r = torch.zeros(4, dtype=torch.float64)
moved_far = 0.0
n_far_r = 0.0
tot = 0.0
seen = set()

for n, (sc, front) in enumerate(samples):
    key = (sc, front)
    if key in seen:                               # the index can repeat a frame across views
        continue
    seen.add(key)
    gt, mask = D._load_depth(f"{D.ROOT}/{sc}/erp_depth/erp_depth_{front:03d}.npy")
    gt = gt[None].float()                         # (1, 1, 256, 512)
    m = mask[None].float() > 0
    fc = T.face_cos_like(gt)
    d_p = (gt * MAX_DEPTH).clamp(0.05, MAX_DEPTH)
    d_r = (gt * MAX_DEPTH / fc).clamp(0.05, MAX_DEPTH)
    ip = torch.bucketize(d_p[m], EDGES)
    ir = torch.bucketize(d_r[m], EDGES)
    cnt_p += torch.bincount(ip, minlength=4).double()
    cnt_r += torch.bincount(ir, minlength=4).double()
    far_r = ir == 3
    n_far_r += far_r.sum().item()
    moved_far += (far_r & (ip == 2)).sum().item()
    tot += m.sum().item()
    if n % 500 == 0:
        print(f"  {n}/{len(samples)} unique={len(seen)}", flush=True)

fp = (cnt_p / tot).numpy()
fr = (cnt_r / tot).numpy()
w = np.array(json.load(open(T.BIN_DIST_PATH))["invfreq_weight_train"])

print(f"\n[data] {len(seen)} unique frames, {int(tot)} valid pixels\n")
print(f"{'bin':>9s} {'E145 planar':>12s} {'E146 radial':>12s} {'invfreq w':>10s} {'E146 mass':>10s}")
for k in range(4):
    print(f"{NAMES[k]:>9s} {fp[k]*100:11.4f}% {fr[k]*100:11.4f}% {w[k]:10.4f} {fr[k]*w[k]:10.4f}")
print(f"\nradial >4m pixels the E145 (planar) expression binned as 1.5-4: "
      f"{100*moved_far/max(n_far_r,1):.2f}%")

gp, gr = fp[3] * 100, fr[3] * 100
ok_p = abs(gp - REF["planar"]) <= TOL
ok_r = abs(gr - REF["radial"]) <= TOL
print(f"\n[I1'] planar control: ref {REF['planar']:.4f}%  got {gp:.4f}%  "
      f"diff {abs(gp-REF['planar']):.4f} pp  {'PASS' if ok_p else 'FAIL'}")
print(f"[I1'] radial treated: ref {REF['radial']:.4f}%  got {gr:.4f}%  "
      f"diff {abs(gr-REF['radial']):.4f} pp  {'PASS' if ok_r else 'FAIL'}")
ok = ok_p and ok_r
print(f"[I1'] {'PASS' if ok else 'FAIL'}")

out = {"n_frames": len(seen), "n_pixels": int(tot), "bin_names": list(NAMES),
       "frac_planar_e145": fp.tolist(), "frac_radial_e146": fr.tolist(),
       "invfreq_weight": w.tolist(), "mass_e146": (fr * w).tolist(),
       "radial_far_binned_as_mid_by_planar_frac": moved_far / max(n_far_r, 1),
       "ref": REF, "tol_pp": TOL, "got_planar_pct": float(gp), "got_radial_pct": float(gr),
       "planar_pass": bool(ok_p), "radial_pass": bool(ok_r), "I1_pass": bool(ok)}
with open("/root/local1/changwoo/e146/results/i1_bin_check_full.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved results/i1_bin_check_full.json")
sys.exit(0 if ok else 1)
