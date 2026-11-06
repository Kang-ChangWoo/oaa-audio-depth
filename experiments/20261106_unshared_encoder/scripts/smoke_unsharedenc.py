"""Smoke for --unshared-enc. Run in BOTH trees (patched repo and the pre-patch copy) and diff.

usage: python _smoke_unsharedenc.py <out.json> [--full]
  writes {case: sha256(output)} for the weight-SHARED path (switch off) so the two trees can be
  compared byte for byte, plus the on-switch structure tests (patched tree only; the reference
  tree skips them because the kwarg does not exist there).
"""
import hashlib
import json
import sys

import torch

torch.set_num_threads(8)
sys.path.insert(0, ".")
from model.oaa import OAAv2Depth  # noqa: E402

OUT = sys.argv[1]
FULL = "--full" in sys.argv
SMALL = dict(enc_res=(64, 128), lh=16, lw=32)      # 2 encoder stages, same code path, CPU-cheap
BIG = dict(enc_res=(256, 512), lh=16, lw=32)       # the released config


def h(t):
    return hashlib.sha256(t.detach().contiguous().float().numpy().tobytes()).hexdigest()[:16]


def run_case(nv, ic, res, drop=None, unshared=False, seed=1234):
    torch.manual_seed(seed)
    kw = dict(res)
    if unshared:
        kw["unshared_enc"] = True
    m = OAAv2Depth(nviews=nv, in_ch=ic, **kw).eval()
    torch.manual_seed(7)
    H, W = res["enc_res"]
    spec = torch.randn(1, nv * ic, H, W)
    ov = None
    if drop is not None:
        ov = torch.ones(1, nv)
        ov[0, drop] = 0.0
    with torch.no_grad():
        out = m(spec, obs_valid=ov)
    return m, spec, ov, out


def params(m):
    return sum(p.numel() for p in m.parameters())


res = {"cases": {}, "notes": []}

# ---- A: weight-shared path (switch off) over every pairfeat input shape, with and without a drop
CASES = [("nv8_ic1", 8, 1), ("nv8_ic2", 8, 2), ("nv8_ic3", 8, 3), ("nv8_ic4", 8, 4),
         ("nv8_ic8", 8, 8), ("nv1_ic8", 1, 8), ("nv4_ic2", 4, 2), ("nv2_ic1", 2, 1)]
for name, nv, ic in CASES:
    for tag, drop in (("", None), ("_drop", 2 if nv > 2 else (0 if nv == 2 else None))):
        m, _, _, out = run_case(nv, ic, SMALL, drop=drop)
        res["cases"][f"small_{name}{tag}"] = {"sha": h(out), "params": params(m)}
if FULL:
    for name, nv, ic in (("nv8_ic1", 8, 1), ("nv8_ic2", 8, 2)):
        m, _, _, out = run_case(nv, ic, BIG)
        res["cases"][f"big_{name}"] = {"sha": h(out), "params": params(m)}

# ---- B..D: switch on (patched tree only)
try:
    OAAv2Depth(nviews=2, in_ch=1, unshared_enc=True, **SMALL)
except TypeError as e:
    res["notes"].append(f"unshared_enc kwarg absent (reference tree): {e}")
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["cases"], indent=1))
    sys.exit(0)

on = {}
# B: forward works, token grid and pose list unchanged
m_u, spec_u, _, out_u = run_case(8, 1, SMALL, unshared=True)
m_s, _, _, out_s = run_case(8, 1, SMALL)
with torch.no_grad():
    t_u, poses_u, _, fine_u = m_u._encode(spec_u)
    t_s, poses_s, _, fine_s = m_s._encode(spec_u)
on["B_forward_shape"] = list(out_u.shape) == list(out_s.shape)
on["B_tokens_total"] = int(t_u.shape[1] * t_u.shape[2])          # nv * M, must equal shared path
on["B_tokens_match"] = list(t_u.shape) == list(t_s.shape) and list(fine_u.shape) == list(fine_s.shape)
on["B_poses"] = len(poses_u) == len(poses_s) == 8

# C: sharing really broken -- perturb the encoder owning observation 3 only
with torch.no_grad():
    for p in m_u.enc_extra[2].parameters():
        p.add_(0.05)
    t_u2, _, _, _ = m_u._encode(spec_u)
delta = (t_u2 - t_u).abs().amax(dim=(0, 2, 3)).tolist()           # per observation
on["C_delta_per_obs"] = [round(d, 6) for d in delta]
on["C_only_obs3_moves"] = (delta[3] > 0) and all(d == 0.0 for i, d in enumerate(delta) if i != 3)
with torch.no_grad():                                             # control: shared path moves ALL
    for p in m_s.enc.parameters():
        p.add_(0.05)
    t_s2, _, _, _ = m_s._encode(spec_u)
d_s = (t_s2 - t_s).abs().amax(dim=(0, 2, 3)).tolist()
on["C_shared_all_move"] = all(d > 0 for d in d_s)

# D: parameter accounting
m_u2, _, _, _ = run_case(8, 1, SMALL, unshared=True)
m_s2, _, _, _ = run_case(8, 1, SMALL)
enc_p = sum(p.numel() for p in m_s2.enc.parameters())
on["D_total_shared"] = params(m_s2)
on["D_total_unshared"] = params(m_u2)
on["D_enc_params"] = enc_p
on["D_matches_formula"] = params(m_u2) == params(m_s2) + 7 * enc_p
on["D_state_dict_extra_keys"] = len([k for k in m_u2.state_dict() if k.startswith("enc_extra.")])

# E: a reference (switch-off) checkpoint still loads strict into a switch-off model
sd = m_s2.state_dict()
m_s3 = OAAv2Depth(nviews=8, in_ch=1, **SMALL)
m_s3.load_state_dict(sd, strict=True)
on["E_reference_ckpt_strict_load"] = True
# and an unshared checkpoint round-trips into an unshared model
m_u3 = OAAv2Depth(nviews=8, in_ch=1, unshared_enc=True, **SMALL)
m_u3.load_state_dict(m_u2.state_dict(), strict=True)
on["E_unshared_ckpt_strict_load"] = True

# F: ckpt.build() passes the flag through (what scoring/resume uses)
try:
    import inspect

    from core import ckpt as ckptmod
    src = inspect.getsource(ckptmod.build)
    on["F_ckpt_forwards_flag"] = "unshared_enc=args.get(\"unshared_enc\"" in src
except Exception as e:  # noqa: BLE001
    on["F_ckpt_forwards_flag"] = f"error: {e}"

# G: one backward + optimizer step on CPU at the small res (does autograd hold)
m_g = OAAv2Depth(nviews=8, in_ch=1, unshared_enc=True, **SMALL)
opt = torch.optim.AdamW(m_g.parameters(), lr=1e-4)
torch.manual_seed(7)
sp = torch.randn(1, 8, 64, 128)
loss = m_g(sp).mean()
loss.backward()
n_nograd = [n for n, p in m_g.named_parameters() if p.grad is None]
opt.step()
on["G_backward_ok"] = bool(torch.isfinite(loss))
on["G_params_without_grad"] = n_nograd[:5]
on["G_n_params_without_grad"] = len(n_nograd)

res["on_switch"] = on
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
