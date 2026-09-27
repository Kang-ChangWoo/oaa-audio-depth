#!/usr/bin/env python3
"""E146 integrity check I1: does the TRAINER's own data path now bin on radial metres?

Loads the data module exactly as train_oaa_e146.py does (core.data.get_data_module, DATA_MODULE and
E135_CUE from the env), pulls N train batches, and recomputes both the E145 expression and the E146
expression on the SAME tensors:

  E145:  d_m = (gt * max_depth).clamp(0.05, max_depth)                     # planar
  E146:  d_m = (gt * max_depth / face_cos_like(gt)).clamp(0.05, max_depth) # radial

`face_cos_like` is imported FROM train_oaa_e146.py, so this checks the shipped function, not a
copy of it. Also reports the invfreq weight actually attached to each bin by the weight file the
trainer will read, and the resulting loss mass per bin.

Pass condition (PREREG_E146 section 4, I1): the E146 `>4 m` pixel fraction must sit near the radial
train-split value 4.15 %, not the planar 2.78 % (tolerance +-0.3 pp for the sample size).
"""
import os, sys, json, importlib
import numpy as np
import torch

sys.path.insert(0, "/root/storage/e146_code")
import train_oaa_e146 as T                       # noqa: E402  (brings face_cos_like + BIN_DIST_PATH)
from core.data import get_data_module            # noqa: E402

N_BATCH = int(os.environ.get("E146_CHECK_BATCHES", "40"))
BS = int(os.environ.get("E146_CHECK_BS", "8"))
MAX_DEPTH = 10.0
EDGES = torch.tensor([0.5, 1.5, 4.0])
NAMES = ("<0.5", "0.5-1.5", "1.5-4", ">4")

DM = get_data_module()
print(f"[cfg] DATA_MODULE={os.environ.get('DATA_MODULE')} E135_CUE={os.environ.get('E135_CUE')} "
      f"module={getattr(DM, '__name__', DM)}", flush=True)
print(f"[cfg] BIN_DIST_PATH={T.BIN_DIST_PATH}  ERP_CONV={T.ERP_CONV}", flush=True)

ds = DM.RotSet("train", "r2")
dl = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=False, num_workers=4)

cnt_p = torch.zeros(4, dtype=torch.float64)
cnt_r = torch.zeros(4, dtype=torch.float64)
moved_far = 0.0        # radial >4 that planar called 1.5-4
n_far_r = 0.0
tot = 0.0

for i, b in enumerate(dl):
    if i >= N_BATCH:
        break
    gt = b["depth"].float()
    m = b["mask"].float() > 0
    fc = T.face_cos_like(gt)
    d_p = (gt * MAX_DEPTH).clamp(0.05, MAX_DEPTH)                 # E145 expression
    d_r = (gt * MAX_DEPTH / fc).clamp(0.05, MAX_DEPTH)            # E146 expression
    ip = torch.bucketize(d_p[m], EDGES)
    ir = torch.bucketize(d_r[m], EDGES)
    cnt_p += torch.bincount(ip, minlength=4).double()
    cnt_r += torch.bincount(ir, minlength=4).double()
    far_r = ir == 3
    n_far_r += far_r.sum().item()
    moved_far += (far_r & (ip == 2)).sum().item()
    tot += m.sum().item()
    if i == 0:
        print(f"[shape] gt={tuple(gt.shape)} face_cos={tuple(fc.shape)} "
              f"1/fc range=[{float((1/fc).min()):.5f}, {float((1/fc).max()):.5f}]", flush=True)

fp = (cnt_p / tot).numpy()
fr = (cnt_r / tot).numpy()
wj = json.load(open(T.BIN_DIST_PATH))
w = np.array(wj["invfreq_weight_train"])

print(f"\n[data] {int(tot)} valid pixels over {min(N_BATCH, len(dl))} batches of {BS}\n")
print(f"{'bin':>9s} {'E145 planar':>12s} {'E146 radial':>12s} {'invfreq w':>10s} {'E146 mass':>10s}")
for k in range(4):
    print(f"{NAMES[k]:>9s} {fp[k]*100:11.4f}% {fr[k]*100:11.4f}% {w[k]:10.4f} {fr[k]*w[k]:10.4f}")

print(f"\nradial >4m pixels that the E145 (planar) expression binned as 1.5-4: "
      f"{100*moved_far/max(n_far_r,1):.2f}%")

TARGET_R, TARGET_P, TOL = 4.1542, 2.7820, 0.30     # train-split 256x512 values, gate0_bin_dist_radial
got = fr[3] * 100
print(f"\n[I1] >4m fraction: planar ref {TARGET_P:.4f}%  radial ref {TARGET_R:.4f}%  got {got:.4f}%")
ok = abs(got - TARGET_R) <= TOL
print(f"[I1] {'PASS' if ok else 'FAIL'} -- |got - radial_ref| = {abs(got-TARGET_R):.4f} pp (tol {TOL} pp)")

out = {"n_pixels": int(tot), "n_batches": min(N_BATCH, len(dl)), "batch_size": BS,
       "bin_names": list(NAMES), "frac_planar_e145": fp.tolist(), "frac_radial_e146": fr.tolist(),
       "invfreq_weight": w.tolist(), "mass_e146": (fr * w).tolist(),
       "bin_dist_path": T.BIN_DIST_PATH, "erp_conv": T.ERP_CONV,
       "radial_far_binned_as_mid_by_planar_frac": moved_far / max(n_far_r, 1),
       "I1_target_radial_pct": TARGET_R, "I1_tol_pp": TOL, "I1_got_pct": float(got), "I1_pass": bool(ok)}
os.makedirs("/root/local1/changwoo/e146/results", exist_ok=True)
with open("/root/local1/changwoo/e146/results/i1_bin_check.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved results/i1_bin_check.json")
sys.exit(0 if ok else 1)
