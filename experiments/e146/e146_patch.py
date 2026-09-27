#!/usr/bin/env python3
"""E146 patch: move the loss-weight BIN computation from planar depth to RADIAL depth.

Base: train_oaa_e145.py  md5 39debb9beb863a1bde02316cbce8670c
Out:  train_oaa_e146.py

Every replacement is a literal string asserted to match EXACTLY ONCE (the E144/E145 pattern), so a
silent drift in the base file aborts the patch instead of producing a wrong trainer.

WHAT CHANGES (and nothing else)
  1. import of EchoRecon's ray_dirs/face_cos + a cached FACE_COS tensor builder
  2. BIN_DIST_PATH default -> gate0_bin_dist_radial.json (the histogram recomputed in radial metres)
  3. training loop: weight_fn now receives radial metres, not planar metres
  4. quick_val: the `obj` weight now receives radial metres, not planar metres
  5. the [E143] banner prints the bin-space + the weight file, so every log proves which it ran

WHAT DOES NOT CHANGE
  * the loss TARGET stays planar `gt` -- the model still predicts what it always predicted
  * optimizer / cosine schedule / EMA / split / augmentation / RNG consumption
  * `val_mae_m` (selects best.pth) -- no distance weight in it at all, so gate G1 is a real test
  * ckpt bundle format, --resume auto, atomic saves, best/best_own dual save
"""
import hashlib, os, sys

SRC = "/root/storage/e146_code/train_oaa_e145.py"
DST = "/root/storage/e146_code/train_oaa_e146.py"
EXPECT_MD5 = "39debb9beb863a1bde02316cbce8670c"

src = open(SRC).read()
got = hashlib.md5(src.encode()).hexdigest()
assert got == EXPECT_MD5, f"base train_oaa_e145.py md5 {got} != {EXPECT_MD5}"
print(f"[ok] base md5 {got}")

REPL = []

# ---- 1. imports + FACE_COS builder -------------------------------------------------------------
REPL.append((
"""from core.data import get_data_module
from core.metrics import cos_lat
from model.oaa import OAAv2Depth
""",
"""from core.data import get_data_module
from core.metrics import cos_lat
from model.oaa import OAAv2Depth

import importlib.util as _ilu                     # E146: load EchoRecon's erp.py by PATH

# ---- E146: the loss-weight bin space ------------------------------------------------------------
# The stored GT (data_0422._load_depth reads erp_depth/) is per-face cubemap z-depth (PLANAR).
# e114_eval.py bins the RADIAL range: EchoRecon/src/erp.py to_radial(d, dirs, "face") = d / face_cos.
# E143/E144/E145 fed the PLANAR metres straight into torch.bucketize, so the training bins and the
# evaluation bins were different sets of pixels. E146 converts to radial before bucketize, so the
# treated set and the measured set agree. `face_cos` is IMPORTED from EchoRecon, not reimplemented.
#
# Loaded by file path rather than sys.path.insert: EchoRecon/src also holds data.py / predict.py /
# eval_fusion.py, and putting that directory on sys.path could shadow a hear360 module later in the
# run. erp.py imports numpy only, so a path load is safe and has no side effects.
_ERP_PATH = os.environ.get("ECHORECON_ERP",
                           "/root/storage/implementation/shared_audio/EchoRecon/src/erp.py")
_erp_spec = _ilu.spec_from_file_location("echorecon_erp", _ERP_PATH)
_erp = _ilu.module_from_spec(_erp_spec)
_erp_spec.loader.exec_module(_erp)
_ray_dirs, _face_cos = _erp.ray_dirs, _erp.face_cos

# e114_eval.py:82 builds dirs at the PREDICTION resolution with convention "right0"; the trainer's
# GT tensor is 256x512 (data_0422.H, data_0422.W), so this is the same grid the eval measures on.
# face_cos is convention-invariant (max|diff| 4.4e-16 across all four CONVENTIONS) -- verified.
ERP_CONV = os.environ.get("E146_ERP_CONV", "right0")
_FC_CACHE = {}


def face_cos_like(t):
    \"\"\"(1, 1, H, W) cos(ray, face-normal) on t's grid/device/dtype. radial = planar / face_cos.\"\"\"
    h, w = t.shape[-2], t.shape[-1]
    key = (h, w, t.device, t.dtype)
    if key not in _FC_CACHE:
        fc = _face_cos(_ray_dirs(h, w, ERP_CONV))
        _FC_CACHE[key] = torch.from_numpy(np.ascontiguousarray(fc)).to(t.device, t.dtype).view(1, 1, h, w)
    return _FC_CACHE[key]
"""))

# ---- 2. weight histogram in radial space -------------------------------------------------------
REPL.append((
"""BIN_DIST_PATH = os.environ.get("E143_BIN_DIST", "/root/storage/e143_code/gate0_bin_dist.json")""",
"""# E146: the invfreq weights must be counted in the SAME space the bins are now computed in,
# otherwise the treatment (radial bins) and the weights (planar histogram) disagree.
BIN_DIST_PATH = os.environ.get("E143_BIN_DIST", "/root/storage/e146_code/gate0_bin_dist_radial.json")"""))

# ---- 3. training loop: radial metres into weight_fn ---------------------------------------------
REPL.append((
"""            d_m = (gt.float() * a.max_depth).clamp(0.05, a.max_depth)   # E114: real-metre range for weighting
            wgt = weight_fn(d_m) * mask""",
"""            # E146: bin on RADIAL metres (what e114_eval.py measures), not planar cubemap z-depth.
            # The loss TARGET below is still planar `gt` -- only the per-pixel weight moves.
            d_m = (gt.float() * a.max_depth / face_cos_like(gt)).clamp(0.05, a.max_depth)
            wgt = weight_fn(d_m) * mask"""))

# ---- 4. quick_val: radial metres into the own-objective weight ----------------------------------
REPL.append((
"""        # same clamp as the training loss: d_m = (gt * max_depth).clamp(0.05, max_depth)
        dw = weight_fn(gt.clamp(0.05, max_depth)) if weight_fn is not None else torch.ones_like(gt)""",
"""        # same clamp AND same bin space as the E146 training loss:
        #   d_m = (gt * max_depth / face_cos).clamp(0.05, max_depth)
        dw = (weight_fn((gt / face_cos_like(gt)).clamp(0.05, max_depth))
              if weight_fn is not None else torch.ones_like(gt))"""))

# ---- 5. banner proves the bin space in every run log --------------------------------------------
REPL.append((
"""        print(f"[E143] cue={a.cue} loss_weight={a.loss_weight} in_ch={a.in_ch} spec_ch={a.nviews * a.in_ch} \"""",
"""        print(f"[E146] bin_space=radial erp_conv={ERP_CONV} bin_dist={BIN_DIST_PATH}", flush=True)
        print(f"[E143] cue={a.cue} loss_weight={a.loss_weight} in_ch={a.in_ch} spec_ch={a.nviews * a.in_ch} \""""))

out = src
for i, (old, new) in enumerate(REPL, 1):
    n = out.count(old)
    assert n == 1, f"replacement {i}: matched {n} times, expected exactly 1\n---\n{old[:200]}"
    out = out.replace(old, new)
    print(f"[ok] replacement {i}: 1 match")

open(DST, "w").write(out)
print(f"[ok] wrote {DST}  md5 {hashlib.md5(out.encode()).hexdigest()}")
