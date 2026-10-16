"""Equivalence gate for the `archfix_` architecture repairs (PREREG_archfix_20261016 section 2).

Run either way:
    python tests/test_oaa_arch_fixes.py        # prints every measured number
    pytest -q tests/test_oaa_arch_fixes.py

Items under test, all of them construction-time and default-off:
    B1  --fine-layerscale   LayerScale on the fine decoder skip instead of zero-init
    B2  --registers         register tokens in the ray cross-attention
    B3  --scene-registers    those registers carry a pooled scene code
    B4  --geo-basis sh4      degree-4 real spherical harmonics for the geometry bias
    C1  (no switch)          batched `_geo_feats`; speed only, gated by bit-identity

Everything that is an *equivalence* claim runs on CPU in float64. On GPU the TF32/flash kernels are
not bit-reproducible, so a CUDA run can only establish a loose bound. Gradient differences are
divided by the LARGEST gradient in the configuration, never by each tensor's own max -- dividing an
eps-level difference by its own ~1e-12 magnitude manufactured a false 8.9e-4 failure in the
`attnfus_` round.

`OAA_REF_MODULE_EDA3871` points at a pristine `eda3871` copy of model/oaa.py (default
`/root/local1/changwoo/archfix_20261016/ref/oaa_ref_eda3871.py`, produced with
`git show eda3871:model/oaa.py`). Missing -> the parity tests FAIL rather than silently pass, unless
OAA_REF_OPTIONAL=1 records an explicit, counted skip.

`ARCHFIX_CKPT` points at a run directory whose best.pth must strict-load into the default-off model
(default the hop-16 control `relevel_h16_stem1_r2_rep_s0`).
"""
import importlib.util
import math
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import oaa as M                                              # noqa: E402
from model.oaa import (OAAv2Depth, RayMicAttn, GEO_FEAT_DIM, _dir_pe,   # noqa: E402
                       _sh4_basis, _SH4_DIM, IDENTITY_INIT_PARAM_SUFFIXES)

REF_PATH = os.environ.get("OAA_REF_MODULE_EDA3871",
                          "/root/local1/changwoo/archfix_20261016/ref/oaa_ref_eda3871.py")
CKPT_DIR = os.environ.get(
    "ARCHFIX_CKPT",
    "/root/local1/changwoo/relevel_hop16_obs48_20261015/runs/rep/relevel_h16_stem1_r2_rep_s0")
SMALL = dict(C=64, lh=4, lw=8, enc_res=(16, 32))     # 256x512 output, tiny attention
PARAM_CEILING_M = 15.45                              # PREREG hard ceiling; the paper's #P prints 15.4
NOTES = []


class _Skipped(Exception):
    """An explicitly requested, counted skip. Never silent: main() reports it and exits non-zero."""


def note(k, v):
    NOTES.append((k, v))
    print(f"    {k} = {v}")


def _maxdiff(a, b):
    return float((a.detach() - b.detach()).abs().max())


def _unit_grid(lh, lw):
    """model.oaa._dir_pe's ray grid rebuilt in float64. The model's own buffer is float32, so its
    rows are unit only to ~1e-7 and the SH identities below would be checked at the wrong precision.
    Same formula as _dir_pe: d(A=0) = +z, d(+pi/2) = +x, elevation on y."""
    el = (torch.arange(lh, dtype=torch.float64) + 0.5) / lh * math.pi - math.pi / 2
    az = (torch.arange(lw, dtype=torch.float64) + 0.5) / lw * 2 * math.pi - math.pi
    E, A = torch.meshgrid(el, az, indexing="ij")
    return torch.stack([torch.cos(E) * torch.sin(A), torch.sin(E),
                        torch.cos(E) * torch.cos(A)], -1).reshape(lh * lw, 3)


def _ref_module():
    if not os.path.exists(REF_PATH):
        msg = (f"pristine eda3871 module not found at {REF_PATH}.\n"
               f"    git show eda3871:model/oaa.py > {REF_PATH}\n"
               f"    or set OAA_REF_OPTIONAL=1 for an explicit, counted skip.")
        if os.environ.get("OAA_REF_OPTIONAL") == "1":
            raise _Skipped(msg)
        raise FileNotFoundError(msg)
    spec = importlib.util.spec_from_file_location("oaa_ref_eda3871", REF_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _small_model(nviews, ref=None, **kw):
    torch.manual_seed(0)
    cls = (ref or M).OAAv2Depth
    return cls(nviews=nviews, **SMALL, **kw).double().eval()


def _ray_mod(dk=8, heads=8, ref=None, **kw):
    torch.manual_seed(0)
    cls = (ref or M).RayMicAttn
    return cls(dk * heads, heads=heads, **kw).double().eval()


def _spec(B, nviews, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, nviews, 24, 40, generator=g, dtype=torch.float64)


def _ray_inputs(B, R, N, Mtok, C, seed=1, scale=1.0, unit=False):
    g = torch.Generator().manual_seed(seed)
    q_in = torch.randn(B, R, C, generator=g, dtype=torch.float64) * scale
    tokens = torch.randn(B, N * Mtok, C, generator=g, dtype=torch.float64) * scale
    el = (torch.arange(R, dtype=torch.float64) + 0.5) / R * math.pi - math.pi / 2
    dir3 = torch.stack([torch.cos(el), torch.sin(el), torch.cos(el) * 0.5], -1)
    if unit:                                   # the real model's rays are unit vectors
        dir3 = dir3 / dir3.norm(dim=-1, keepdim=True)
    poses = [(2 * math.pi * i / max(N // 2, 1), (-1.0, 1.0)[i % 2]) for i in range(N)]
    return q_in, tokens, dir3, poses


def _copy_shared(src, dst):
    """Copy every parameter/buffer the two modules have in common (new options add keys only)."""
    s, d = src.state_dict(), dst.state_dict()
    out = dst.load_state_dict({k: v for k, v in s.items() if k in d}, strict=False)
    assert not out.unexpected_keys, out.unexpected_keys
    return sorted(set(d) - set(s))


def _grads_of(m, x, named_inputs=False):
    m.zero_grad()
    xx = x.clone().requires_grad_(True) if named_inputs else x
    D = m(xx)
    (D * torch.linspace(0.3, 1.7, D.numel(), dtype=D.dtype).view_as(D)).sum().backward()
    g = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    if named_inputs:
        g["input.spec"] = xx.grad.clone()
    return D.detach(), g


def _cmp_grads(ga, gb, tag):
    """max relative gradient difference, scaled by the largest gradient in the configuration."""
    assert ga.keys() == gb.keys(), (tag, sorted(set(ga) ^ set(gb)))
    scale = max(float(v.abs().max()) for v in ga.values())
    assert scale > 0, f"{tag}: every gradient is zero, the comparison would be vacuous"
    worst, where = 0.0, None
    for k in ga:
        d = _maxdiff(ga[k], gb[k]) / scale
        if d > worst:
            worst, where = d, k
    return worst, where, scale


# ======================================================================= 1. switches off == eda3871
def test_all_switches_off_is_bit_identical_to_eda3871():
    """Every new switch off must be the pristine eda3871 model op for op: forward AND every
    parameter gradient, max|diff| exactly 0.0, in float64, at N = 2, 4, 8."""
    ref = _ref_module()
    worst_f = worst_g = 0.0
    for nv in (2, 4, 8):
        old = _small_model(nv, ref=ref).train()
        new = _small_model(nv).train()
        assert set(old.state_dict()) == set(new.state_dict()), f"N={nv}: default key sets differ"
        new.load_state_dict(old.state_dict(), strict=True)
        x = _spec(2, nv)
        Do, go = _grads_of(old, x, named_inputs=True)
        Dn, gn = _grads_of(new, x, named_inputs=True)
        df = _maxdiff(Do, Dn)
        dg, where, scale = _cmp_grads(go, gn, f"N={nv}")
        worst_f, worst_g = max(worst_f, df), max(worst_g, dg)
        assert df == 0.0, f"N={nv}: forward differs by {df:.3e}"
        assert dg == 0.0, f"N={nv}: gradient differs by {dg:.3e} at {where} (scale {scale:.3e})"
        assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    note("off_vs_eda3871_forward_maxdiff", f"{worst_f:.3e} (float64, N=2/4/8, must be exactly 0)")
    note("off_vs_eda3871_grad_maxrel", f"{worst_g:.3e} (scaled by the largest gradient in the "
                                       f"configuration, must be exactly 0)")


def test_geofast_is_bit_identical_to_the_loop():
    """C1: the batched `_geo_feats` vs eda3871's per-observation Python loop, forward and the
    bias_mlp / Q / K / V gradients through RayMicAttn. Exactly 0.0, float64 AND float32, on the real
    16x32 ray grid and on a non-unit random one."""
    ref = _ref_module()
    worst_f = worst_g = 0.0
    rows = []
    for dt in (torch.float64, torch.float32):
        for N in (2, 4, 8):
            ro = _ray_mod(ref=ref).to(dt).eval()
            no = _ray_mod().to(dt).eval()
            no.load_state_dict(ro.state_dict(), strict=True)
            # the production ray grid (unit), and a deliberately non-unit set
            grids = {"grid16x32": _dir_pe(16, 32, torch.device("cpu")).to(dt)[:, :3]}
            q, tk, d3, poses = _ray_inputs(2, 16, N, 8, 64)
            grids["nonunit"] = d3.to(dt)
            for tag, rays in grids.items():
                R = rays.shape[0]
                qq, tkk, _, _ = _ray_inputs(2, R, N, 8, 64)
                # the raw feature tensor first: the claim C1 actually makes
                fo = ro._geo_feats(rays, poses)            # eda3871 returns a per-observation list
                fo = torch.stack(fo, 1) if isinstance(fo, list) else fo
                fn = no._geo_feats(rays, poses)
                dfeat = _maxdiff(fo, fn)
                assert dfeat == 0.0, (dt, N, tag, dfeat)
                outs, grads = [], []
                for mod in (ro, no):
                    mod.zero_grad()
                    a, b = qq.to(dt).clone().requires_grad_(True), tkk.to(dt).clone().requires_grad_(True)
                    o = mod(a, b, rays, poses, 8)
                    o.square().sum().backward()
                    outs.append(o.detach())
                    grads.append({"input.q": a.grad, "input.tokens": b.grad,
                                  **{n: p.grad for n, p in mod.named_parameters() if p.grad is not None}})
                df = _maxdiff(*outs)
                dg, where, _ = _cmp_grads(*grads, f"{dt} N={N} {tag}")
                worst_f, worst_g = max(worst_f, df), max(worst_g, dg)
                rows.append((str(dt).split(".")[-1], N, tag, dfeat, df, dg))
                assert df == 0.0, (dt, N, tag, df)
                assert dg == 0.0, (dt, N, tag, dg, where)
    note("geofast_feats_maxdiff", "0.0 for every (dtype, N, grid): 12 configurations")
    note("geofast_forward_maxdiff", f"{worst_f:.3e}")
    note("geofast_grad_maxrel", f"{worst_g:.3e}")
    # and the speed claim's mechanism: one bias_mlp call and no per-observation host->device tensor
    m = _ray_mod()
    d3 = _dir_pe(4, 8, torch.device("cpu"))[:, :3].double()
    poses = [(0.0, -1.0), (0.0, 1.0)]
    m._geo_feats(d3, poses)
    key_a = m._pose_const[0]
    m._geo_feats(d3, poses)
    assert m._pose_const[0] is key_a or m._pose_const[0] == key_a, "pose constants must be cached"
    m._geo_feats(d3, [(1.0, -1.0), (1.0, 1.0)])
    assert m._pose_const[0] != key_a, "a different pose list must invalidate the cache"
    assert "_pose_const" not in m.state_dict(), "the cache must never become state"
    note("geofast_pose_cache", "cached per pose list, invalidated on change, absent from state_dict")


# ======================================================================= 2. B1 LayerScale
def test_b1_layerscale_gamma_zero_is_bit_identical():
    """PREREG B1 gate (i) says "at gamma = 0 it must be bit-identical to the reference". Measured,
    that premise is HALF WRONG, and this test records the correction rather than the wish. The
    OUTPUT is bit-identical (max|diff| == 0.0) and so is every gradient EXCEPT the two parameters
    of the added conv itself:

      reference   out = W*x + b, W = b = 0   ->   dL/dW = g (*) x != 0,  dL/db = sum(g) != 0
      gamma = 0   out = 0 * (W'*x + b')      ->   dL/dW' = 0,            dL/db' = 0

    Both branches add exactly zeros to the decoder, so nothing downstream can tell them apart, but
    the gradient ARRIVING at the added conv cannot agree: a zero weight does not kill dL/dW, a zero
    multiplier does. That is a property of gamma, not a defect -- at gamma = 0 the branch is off.
    The exception set is therefore exactly {fine_to_dec.weight, fine_to_dec.bias}, plus `fine_ls`,
    which has no counterpart in the reference at all. It is enumerated in advance rather than
    discovered per run, and both sides of every exception are reported as numbers below."""
    ref = _ref_module()
    worst, worst_bias = 0.0, {}
    for nv in (2, 8):
        old = _small_model(nv, ref=ref).train()
        ls = _small_model(nv, fine_layerscale=0.1).train()
        added = _copy_shared(old, ls)
        assert added == ["fine_ls"], added
        with torch.no_grad():
            ls.fine_ls.zero_()
        x = _spec(2, nv)
        Do, go = _grads_of(old, x)
        Dn, gn = _grads_of(ls, x)
        d = _maxdiff(Do, Dn)
        worst = max(worst, d)
        assert d == 0.0, (nv, d)
        exc = ("fine_to_dec.weight", "fine_to_dec.bias")
        a = {k: v for k, v in go.items() if k not in exc}
        b = {k: v for k, v in gn.items() if k in a}
        dg, where, _ = _cmp_grads(a, b, f"N={nv}")
        assert dg == 0.0, (nv, dg, where)
        assert "fine_ls" not in go, "the reference must not carry a LayerScale parameter"
        # the exception set, enumerated in the docstring and reported rather than hidden
        for k in exc:
            worst_bias[(nv, k)] = (float(go[k].abs().max()), float(gn[k].abs().max()))
            assert worst_bias[(nv, k)][1] == 0.0, (
                f"gamma = 0 must zero the whole branch gradient, {k} got "
                f"{worst_bias[(nv, k)][1]:.3e}")
            assert worst_bias[(nv, k)][0] > 0.0, (
                f"the reference's {k} DOES get a gradient; if it did not, this exception would "
                f"not exist and the gate could have stayed a plain bit-identity")
    note("b1_gamma0_vs_eda3871_maxdiff", f"{worst:.3e} (float64, N=2/8, output must be exactly 0)")
    note("b1_gamma0_grad_exception",
         "every gradient bit-identical EXCEPT fine_to_dec.{weight,bias} -- PREREG premise "
         "corrected, see the docstring. |g| ref vs gamma=0: "
         + ", ".join(f"N={k[0]} {k[1]}: {v[0]:.3e} vs {v[1]:.3e}"
                     for k, v in worst_bias.items()))


def test_b1_layerscale_restores_the_gradient_path():
    """The mechanism claim: with the reference zero-init, `fine_erp`'s producers get a gradient of
    EXACTLY zero at step 0 (grad = W^T g with W = 0). With LayerScale they do not. If this test
    failed, B1 would be solving a problem that does not exist."""
    nv = 2
    fine = ("fine_q", "fine_in.weight", "fine_dir_mlp.0.weight", "fine_ln.weight",
            "fine_lift.in_proj_weight", "fine_lift.out_proj.weight", "fine_tf_pe")
    base = _small_model(nv).train()
    _, gb = _grads_of(base, _spec(2, nv))
    zero = {k: float(gb[k].abs().max()) for k in fine}
    note("b1_zeroinit_fine_branch_grad_absmax", {k: f"{v:.3e}" for k, v in zero.items()})
    assert max(zero.values()) == 0.0, "the zero-init symptom is not reproducible: " + str(zero)
    for gamma in (1e-4, 1e-2, 0.1):
        ls = _small_model(nv, fine_layerscale=gamma).train()
        _, gl = _grads_of(ls, _spec(2, nv))
        live = {k: float(gl[k].abs().max()) for k in fine}
        assert min(live.values()) > 0.0, (gamma, live)
        if gamma == 1e-4:
            note("b1_layerscale_1e-4_fine_branch_grad_absmax",
                 {k: f"{v:.3e}" for k, v in live.items()})
    # gamma itself receives an O(1) gradient, which is how it can grow out of a small init
    ls = _small_model(nv, fine_layerscale=1e-4).train()
    _, gl = _grads_of(ls, _spec(2, nv))
    note("b1_gamma_grad_vs_fine_to_dec_grad",
         f"gamma {float(gl['fine_ls'].abs().max()):.3e} vs fine_to_dec.weight "
         f"{float(gl['fine_to_dec.weight'].abs().max()):.3e}")
    assert float(gl["fine_ls"].abs().max()) > 0.0


def test_b1_refuses_the_incoherent_combination():
    """--fine-layerscale and --fine-skip-init are two different fixes for the same symptom and both
    set the fine-skip conv's init; the combination must raise, not silently pick one."""
    try:
        _small_model(2, fine_layerscale=0.1, fine_skip_init=0.02)
        raise RuntimeError("fine_layerscale + fine_skip_init should have been refused")
    except AssertionError as e:
        assert "fine-layerscale" in str(e), e
    note("b1_conflict_refused", "--fine-layerscale + --fine-skip-init raises")


# ======================================================================= 3. B2 registers
def test_b2_masked_registers_are_bit_identical_to_no_registers():
    """PREREG: `--registers 4` with the register key/value rows masked out must be identical to
    `--registers 0`. Exactly 0.0, forward and every shared gradient, float64, N = 2/4/8."""
    worst_f = worst_g = 0.0
    for nv in (2, 4, 8):
        for nreg in (1, 4, 8):
            off = _small_model(nv).train()
            on = _small_model(nv, registers=nreg).train()
            added = _copy_shared(off, on)
            assert added == ["ray_mic.reg"], added
            x = _spec(2, nv)
            Do, go = _grads_of(off, x)
            try:
                M._MASK_REGISTERS = True
                Dn, gn = _grads_of(on, x)
            finally:
                M._MASK_REGISTERS = False
            df = _maxdiff(Do, Dn)
            dg, where, _ = _cmp_grads(go, {k: v for k, v in gn.items() if k in go}, f"N={nv}")
            worst_f, worst_g = max(worst_f, df), max(worst_g, dg)
            assert df == 0.0, (nv, nreg, df)
            assert dg == 0.0, (nv, nreg, dg, where)
    note("b2_masked_registers_forward_maxdiff", f"{worst_f:.3e} (9 configurations, must be exactly 0)")
    note("b2_masked_registers_grad_maxrel", f"{worst_g:.3e}")


def test_b2_registers_are_not_inert_and_take_mass():
    """Unmasked registers must change the output and must actually receive attention mass, otherwise
    the masked-equivalence test above would be proving nothing."""
    nv = 8
    off = _small_model(nv)
    on = _small_model(nv, registers=4)
    _copy_shared(off, on)
    x = _spec(2, nv)
    d = _maxdiff(off(x), on(x))
    note("b2_register_output_shift", f"{d:.3e} (same weights otherwise; must be > 0)")
    assert d > 1e-9
    rm = _ray_mod(registers=4)
    rm.collect_stats = True
    q, tk, d3, poses = _ray_inputs(2, 16, 8, 8, 64)
    rm(q, tk, d3, poses, 8)
    mass = rm.stats["reg_mass_mean"]
    note("b2_register_mass_at_init", f"{mass:.4f} of the softmax (4 registers vs 64 acoustic tokens)")
    assert 0.0 < mass < 1.0
    # the diagnostic must be reported per head too, so B2's decision can be attributed
    assert rm.stats["reg_mass_per_head"].shape == (8,)
    # and the register gradient is alive
    rm.zero_grad(); rm(q, tk, d3, poses, 8).square().sum().backward()
    assert float(rm.reg.grad.abs().max()) > 0


def test_b2_registers_refuse_hierarchical_fusion():
    """Under `hier` the outer softmax is over observations and a register belongs to none of them.
    PREREG puts that design question out of scope, so the combination must raise."""
    for vf in ("hier", "hier-geo"):
        try:
            _small_model(4, registers=4, view_fusion=vf)
            raise RuntimeError(f"registers + {vf} should have been refused")
        except AssertionError as e:
            assert "register" in str(e), e
    try:
        _small_model(4, registers=4, no_cross=True)
        raise RuntimeError("registers + no_cross should have been refused")
    except AssertionError as e:
        assert "no-cross" in str(e), e
    note("b2_conflicts_refused", "registers + hier / hier-geo / no-cross all raise")


# ======================================================================= 4. B3 scene registers
def test_b3_zero_init_starts_exactly_at_b2():
    """`scene_mlp` is zero-initialised, so scene-conditioned registers must start bit-identical to
    plain registers -- the two cells then differ only in what the optimiser may learn."""
    worst = 0.0
    for nv in (2, 8):
        b2 = _small_model(nv, registers=4)
        b3 = _small_model(nv, registers=4, scene_registers=True)
        added = _copy_shared(b2, b3)
        assert added == ["ray_mic.scene_mlp.bias", "ray_mic.scene_mlp.weight"], added
        x = _spec(2, nv)
        d = _maxdiff(b2(x), b3(x))
        worst = max(worst, d)
        assert d == 0.0, (nv, d)
    note("b3_init_equals_b2_maxdiff", f"{worst:.3e} (must be exactly 0)")
    # not inert once trained, and the scene code is per sample (it must depend on the input)
    b3 = _small_model(8, registers=4, scene_registers=True)
    torch.nn.init.normal_(b3.ray_mic.scene_mlp.weight, std=0.05)
    x = _spec(2, 8)
    assert _maxdiff(_small_model(8, registers=4)(x), b3(x)) > 1e-9
    x2 = x.clone(); x2[0] = torch.rand_like(x2[0]) * 5.0
    out = b3(x)
    out2 = b3(x2)
    note("b3_scene_code_is_per_sample",
         f"sample 1 unchanged: {_maxdiff(out[1], out2[1]):.3e} (a *global* pooled code would "
         f"leak sample 0 into sample 1 only if pooling crossed the batch)")
    assert _maxdiff(out[1], out2[1]) == 0.0, "the scene pooling must not cross the batch dimension"


def test_b3_requires_registers():
    try:
        _small_model(4, scene_registers=True)
        raise RuntimeError("scene_registers without registers should have been refused")
    except AssertionError as e:
        assert "scene-registers" in str(e), e
    note("b3_conflict_refused", "--scene-registers without --registers raises")


# ======================================================================= 5. B4 spherical harmonics
def _scipy_real_sh(v):
    """Degree<=4 real orthonormal SH of unit vectors v (...,3), from scipy -- an INDEPENDENT
    implementation, not a re-run of ours. scipy >= 1.15 renamed sph_harm -> sph_harm_y(n, m, theta,
    phi) with theta the POLAR angle; older scipy has sph_harm(m, n, azim, polar)."""
    import numpy as np
    try:
        from scipy.special import sph_harm_y
        Y = lambda l, m, th, ph: sph_harm_y(l, m, th, ph)
    except ImportError:                                            # scipy < 1.15
        from scipy.special import sph_harm
        Y = lambda l, m, th, ph: sph_harm(m, l, ph, th)
    x, y, z = [np.asarray(v[..., i], dtype=np.float64) for i in range(3)]
    th = np.arccos(np.clip(z, -1.0, 1.0))                          # polar
    ph = np.arctan2(y, x)                                          # azimuth
    cols = []
    for l in range(5):
        for m in range(-l, l + 1):
            c = Y(l, abs(m), th, ph)
            if m == 0:
                r = np.real(c)
            elif m > 0:
                r = math.sqrt(2.0) * (-1) ** m * np.real(c)
            else:
                r = math.sqrt(2.0) * (-1) ** m * np.imag(c)
            cols.append(r)
    return torch.from_numpy(np.stack(cols, -1))


# `_sh4_basis` uses the 3D-Gaussian-Splatting / plenoxels constant table, whose columns are the
# canonical real SH times (-1)^|m| (the Condon-Shortley phase folded into the constants instead of
# left in the basis). That factor is a GAUGE choice, not an error: `bias_mlp`'s first Linear sits
# immediately after the basis and absorbs a per-column sign, so the span, the orthonormality and
# the training trajectory are all unchanged. Keeping the published table and declaring the
# convention is safer than hand-flipping 25 transcribed constants.
#
# The sign vector is a formula fixed in advance and is never fitted to our own output, so a wrong
# normalisation, a wrong polynomial, or two swapped columns still fail the comparison below.
_SH4_SIGN = torch.tensor([(-1.0) ** abs(m) for l in range(5) for m in range(-l, l + 1)],
                         dtype=torch.float64)


def test_b4_sh4_matches_an_independent_implementation():
    """The 25 basis functions must BE the real spherical harmonics of degree <= 4, checked against
    scipy, not against themselves -- up to the declared per-column sign convention of `_SH4_SIGN`
    -- and orthonormal on the sphere, which is a convention-free proof that the basis is complete
    and correctly normalised."""
    v = _unit_grid(64, 128)                                        # unit by construction
    assert float((v.norm(dim=-1) - 1.0).abs().max()) < 1e-15, "the ray grid must be unit vectors"
    # and the model's own float32 buffer is the same grid to float32 precision
    assert _maxdiff(_dir_pe(64, 128, torch.device("cpu"))[:, :3].double(), v) < 1e-6
    ours = _sh4_basis(v)
    assert ours.shape[-1] == _SH4_DIM == 25, ours.shape
    try:
        theirs = _scipy_real_sh(v) * _SH4_SIGN
    except ImportError as e:
        raise _Skipped(f"scipy not importable, cannot check SH against an independent impl: {e}")
    d = _maxdiff(ours, theirs)
    per_l = []
    i = 0
    for l in range(5):
        n = 2 * l + 1
        per_l.append(f"l={l}:{_maxdiff(ours[..., i:i + n], theirs[..., i:i + n]):.1e}")
        i += n
    note("b4_sh4_vs_scipy_maxdiff",
         f"{d:.3e} against scipy x (-1)^|m|, the 3DGS sign convention  ({' '.join(per_l)})")
    assert d < 1e-12, f"sh4 is not the real degree-4 SH basis (max|diff| {d:.3e})"
    # orthonormality: <Y_i, Y_j> over the sphere with the cos-latitude area weight = identity
    lh, lw = 256, 512
    g = _unit_grid(lh, lw)
    el = (torch.arange(lh, dtype=torch.float64) + 0.5) / lh * math.pi - math.pi / 2
    w = torch.cos(el).repeat_interleave(lw)                        # dA = cos(el) dEl dAz
    w = w / w.sum() * 4 * math.pi
    Y = _sh4_basis(g)
    G = (Y * w[:, None]).transpose(0, 1) @ Y
    off = float((G - torch.eye(25, dtype=torch.float64)).abs().max())
    note("b4_sh4_gram_minus_identity_maxabs", f"{off:.3e} (256x512 quadrature; orthonormal basis)")
    assert off < 2e-3, off


def test_b4_sh4_is_the_bias_input_and_nothing_else():
    """geo_basis only widens `bias_mlp`'s first layer: 5 -> 27 inputs (25 SH + ray.ear_axis + ear).
    `raw` must stay bit-identical to eda3871 (covered above) and `sh4` must change the output."""
    assert GEO_FEAT_DIM == {"raw": 5, "sh4": 27}, GEO_FEAT_DIM
    raw = _small_model(4)
    sh = _small_model(4, geo_basis="sh4")
    assert sh.ray_mic.bias_mlp[0].in_features == 27
    assert raw.ray_mic.bias_mlp[0].in_features == 5
    added = sorted(set(sh.state_dict()) - set(raw.state_dict()))
    assert added == [], f"sh4 must add no NEW keys, only widen one: {added}"
    assert sh.state_dict()["ray_mic.bias_mlp.0.weight"].shape == (64, 27)
    # the query-side direction MLPs must keep the 6-dim _dir_pe (PREREG scopes B4 to the bias only)
    for nm in ("dir_mlp", "glob_dir", "fine_dir_mlp"):
        assert getattr(sh, nm)[0].in_features == 6, nm
    x = _spec(2, 4)
    d = _maxdiff(raw(x), sh(x))
    note("b4_sh4_output_shift", f"{d:.3e} (must be > 0: the basis reaches the logits)")
    assert d > 1e-9
    # the first 3 raw columns are exactly what sh4 expands, and the last 2 pass through untouched
    rm = _small_model(2, geo_basis="sh4").ray_mic
    d3 = _dir_pe(4, 8, torch.device("cpu")).double()[:, :3]
    poses = [(0.3, -1.0), (0.3, 1.0)]
    f_raw = rm._geo_feats(d3, poses)
    f_sh = rm._geo_basis_feats(d3, poses)
    assert _maxdiff(f_sh[..., :25], _sh4_basis(f_raw[..., :3])) == 0.0
    assert _maxdiff(f_sh[..., 25:], f_raw[..., 3:]) == 0.0
    note("b4_sh4_passthrough_columns", "ray.ear_axis and ear_sign reach bias_mlp unchanged")


# ======================================================================= 6. parameter budget
def test_parameter_budget_at_the_real_width():
    """PREREG's budget table, measured at the configuration the cells actually train (C=256,
    2 observations, 16x32). The ceiling is hard: 15.45 M."""
    torch.manual_seed(0)
    base = OAAv2Depth(C=256, nviews=2)
    nb = sum(p.numel() for p in base.parameters())
    want = {"B1 ls": 128, "B2 reg": 1024, "B3 sreg": 1024 + 65792, "B4 sh": 1408}
    cfg = {"B1 ls": dict(fine_layerscale=1e-4), "B2 reg": dict(registers=4),
           "B3 sreg": dict(registers=4, scene_registers=True), "B4 sh": dict(geo_basis="sh4")}
    rows = {}
    for k, kw in cfg.items():
        torch.manual_seed(0)
        n = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, **kw).parameters())
        rows[k] = n - nb
        assert rows[k] == want[k], f"{k}: added {rows[k]}, PREREG says {want[k]}"
    torch.manual_seed(0)
    allon = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, fine_layerscale=1e-4, registers=4,
                                              scene_registers=True, geo_basis="sh4").parameters())
    note("param_base_M", f"{nb / 1e6:.4f} (PREREG 15.3540)")
    note("param_added", rows)
    note("param_all_on_M", f"{allon / 1e6:.4f} (PREREG 15.4223, ceiling {PARAM_CEILING_M})")
    assert abs(nb / 1e6 - 15.3540) < 5e-4, nb
    assert abs(allon / 1e6 - 15.4223) < 5e-4, allon
    assert allon / 1e6 <= PARAM_CEILING_M, f"{allon / 1e6:.4f} M breaks the {PARAM_CEILING_M} M ceiling"


def test_param_lr_group_picks_up_scene_mlp_only():
    """`scene_mlp` is zero-initialised, so it joins the --param-lr-mult set; `fine_ls` deliberately
    does NOT (a LayerScale gamma is small but non-zero by design)."""
    m = _small_model(8, registers=4, scene_registers=True, fine_layerscale=1e-4)
    hit = sorted(n for n, _ in m.named_parameters() if n.endswith(IDENTITY_INIT_PARAM_SUFFIXES))
    note("param_lr_group_with_archfix", hit)
    assert hit == ["fine_to_dec.bias", "fine_to_dec.weight",
                   "ray_mic.scene_mlp.bias", "ray_mic.scene_mlp.weight"], hit
    assert not any(n.endswith(IDENTITY_INIT_PARAM_SUFFIXES) for n in ("fine_ls", "ray_mic.reg"))


# ======================================================================= 7. checkpoint contracts
def test_existing_hop16_checkpoint_strict_loads_into_default_off_model():
    """A real trained hop-16 control checkpoint must still load into the default-off model with
    strict=True. This is the claim that every switch creates its parameters only when it is on."""
    import json
    from core.ckpt import build, _stft_from_manifest
    p = os.path.join(CKPT_DIR, "best.pth")
    if not os.path.exists(p):
        raise _Skipped(f"no checkpoint at {p}; set ARCHFIX_CKPT to a scored hop-16 control run")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    a = ck["args"]
    mf = os.path.join(CKPT_DIR, "manifest.json")
    st = (json.load(open(mf)).get("stft") or {}) if os.path.exists(mf) else {}
    DM = types.SimpleNamespace(IN_CH={"r2": 2, "cB": 4, "fb": 4, "r6": 6, "r8": 8}, POSES={},
                               WINDOW=int(st.get("STFT_WINDOW") or 0),
                               HOP=int(st.get("STFT_HOP") or 160), WIN=int(st.get("STFT_WIN") or 400),
                               N_FFT=int(st.get("STFT_NFFT") or 512))
    DM.__name__ = a.get("data_module", "data_0422")
    # a hop-16 run records its STFT in manifest.json, NOT in the checkpoint args, so build()'s
    # guard reads its own default 160 and refuses the run. Proving that here makes this test a
    # regression test for core.ckpt._stft_from_manifest as well.
    try:
        build(a, DM)
        raise AssertionError("build() accepted a hop-16 checkpoint without the manifest STFT; the "
                             "guard that _stft_from_manifest exists for is gone")
    except RuntimeError as e:
        assert "STFT_HOP" in str(e), e
        note("hop16_needs_stft_from_manifest", f"raw args -> {str(e).split(' — ')[0]}")
    m, dmode, nch, kind, poses = build(_stft_from_manifest(CKPT_DIR, a), DM)
    for k in ("fine_layerscale", "registers", "scene_registers", "geo_basis"):
        assert k not in a, f"the control run must predate {k}"
    assert (m.fine_layerscale, m.ray_mic.registers, m.ray_mic.scene_registers,
            m.ray_mic.geo_basis) == (0.0, 0, False, "raw")
    m.load_state_dict(ck["state_dict"], strict=True)
    note("real_ckpt_strict_load", f"{os.path.basename(CKPT_DIR)} ({dmode}, {nch}ch, hop "
                                  f"{DM.HOP}) -> default-off model, strict=True OK")
    # the original symptom B1 targets, read off the checkpoint that motivated it
    sd = ck["state_dict"]
    rms = lambda k: float(sd[k].double().pow(2).mean().sqrt())
    k0 = next(k for k in sd if k.startswith("up_stages.0.") and k.endswith(".weight")
              and sd[k].dim() == 4)
    note("real_ckpt_fine_to_dec_rms_vs_decoder",
         f"fine_to_dec.weight {rms('fine_to_dec.weight'):.4f} vs {k0} {rms(k0):.4f}")


def test_new_configs_round_trip_through_core_ckpt():
    """Save {state_dict, args} -> core.ckpt.build(args, DM) -> strict load -> identical prediction,
    for every archfix configuration, so a trained cell can actually be scored."""
    from core.ckpt import build
    DM = types.SimpleNamespace(IN_CH={"r2": 2, "cB": 4, "fb": 4, "r6": 6, "r8": 8}, POSES={},
                               WINDOW=0, HOP=160, WIN=400, N_FFT=512)
    DM.__name__ = "data_0422"
    cfgs = [{}, {"fine_layerscale": 1e-4}, {"registers": 4},
            {"registers": 4, "scene_registers": True}, {"geo_basis": "sh4"},
            {"fine_layerscale": 1e-4, "registers": 4, "scene_registers": True, "geo_basis": "sh4"},
            {"qk_norm": True, "geo_basis": "sh4"}]
    for extra in cfgs:
        args = dict(nviews=2, data_mode="r2", dim=64, rounds=2, lift_h=4, lift_w=8,
                    stem_stride1=True, max_depth=10.0, dec_deep=True, rounds_wired=True,
                    cond_mode="adaln", full_res_enc=True, multi_scale_lift=True,
                    data_module="data_0422", **extra)
        torch.manual_seed(0)
        src = OAAv2Depth(C=64, nviews=2, lh=4, lw=8, stem_stride1=True,
                         qk_norm=extra.get("qk_norm", False),
                         fine_layerscale=extra.get("fine_layerscale", 0.0),
                         registers=extra.get("registers", 0),
                         scene_registers=extra.get("scene_registers", False),
                         geo_basis=extra.get("geo_basis", "raw")).eval()
        rebuilt, dmode, nch, kind, poses = build(args, DM)
        assert set(rebuilt.state_dict()) == set(src.state_dict()), extra
        rebuilt.load_state_dict(src.state_dict(), strict=True)
        rebuilt.eval()
        x = torch.rand(1, 2, 24, 40)
        with torch.no_grad():
            d = _maxdiff(src(x), rebuilt(x))
        assert d == 0.0, (extra, d)
    note("archfix_ckpt_round_trip", f"{len(cfgs)} configs strict-loaded and prediction-compared: "
                                    f"max|diff| = 0.0")


def test_forward_contract_for_every_cell_config():
    """(B,1,256,512) in [0,1], finite, gradients reaching every new parameter, for the exact
    configurations round 1 queues."""
    cells = {"A1 qk": dict(qk_norm=True), "B1 ls": dict(fine_layerscale=1e-4),
             "B2 reg": dict(registers=4), "B3 sreg": dict(registers=4, scene_registers=True),
             "B4 sh": dict(geo_basis="sh4")}
    newp = {"B1 ls": ["fine_ls"], "B2 reg": ["ray_mic.reg"],
            "B3 sreg": ["ray_mic.reg", "ray_mic.scene_mlp.weight"], "B4 sh": [], "A1 qk": []}
    for nv in (2, 4, 8):
        for name, kw in cells.items():
            m = _small_model(nv, **kw).train()
            D = m(_spec(2, nv))
            assert D.shape == (2, 1, 256, 512) and torch.isfinite(D).all(), (nv, name)
            assert float(D.min()) >= 0.0 and float(D.max()) <= 1.0, (nv, name)
            D.sum().backward()
            pm = dict(m.named_parameters())
            for q in newp[name] + ["ray_mic.q.weight", "ray_mic.bias_mlp.0.weight", "fine_to_dec.weight"]:
                g = pm[q].grad
                assert g is not None and torch.isfinite(g).all(), (nv, name, q)
                if q != "fine_to_dec.weight" or kw.get("fine_layerscale"):
                    assert float(g.abs().sum()) > 0, (nv, name, q)
    note("cell_forward_contract", f"{3 * len(cells)} configurations (N=2/4/8 x 5 cells) "
                                  f"all (2,1,256,512) in [0,1] with live gradients")


def test_obs_mask_still_works_with_registers():
    """Registers must not break the explicit observation masking: an invalid observation still
    cannot change the prediction, and the registers do not smuggle it back in."""
    nv = 8
    m = _small_model(nv, registers=4)
    ov = torch.ones(2, nv, dtype=torch.bool); ov[:, [1, 4, 6]] = False
    x = _spec(2, nv)
    a = m(x, obs_valid=ov)
    x2 = x.clone()
    x2[:, [1, 4, 6]] = torch.rand_like(x2[:, [1, 4, 6]]) * 7.0 + 3.0
    d = _maxdiff(a, m(x2, obs_valid=ov))
    note("registers_obs_mask_invariance", f"{d:.3e}")
    assert d < 1e-12, d
    try:
        M._SHORTCIRCUIT_ALL_VALID = False
        dz = _maxdiff(m(x), m(x, obs_valid=torch.ones(2, nv, dtype=torch.bool)))
    finally:
        M._SHORTCIRCUIT_ALL_VALID = True
    note("registers_allvalid_identity", f"{dz:.3e}")
    assert dz < 1e-10, dz
    # a scene-conditioned register pools over ALL tokens incl. the invalid ones -- record that
    # honestly rather than claim an invariance the implementation does not have
    m3 = _small_model(nv, registers=4, scene_registers=True)
    torch.nn.init.normal_(m3.ray_mic.scene_mlp.weight, std=0.05)
    d3 = _maxdiff(m3(x, obs_valid=ov), m3(x2, obs_valid=ov))
    note("sreg_obs_mask_leak",
         f"{d3:.3e} (UNFAVOURABLE: with a trained scene_mlp the pooled code still sees the "
         f"zeroed-spectrogram tokens of masked observations. Round 1 uses no --subset-aug at "
         f"2 observations, so no cell is affected; it matters only if B3 is ever run at 8 obs.)")


# ======================================================================= runner
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed, skipped = [], []
    for t in tests:
        print(f"[{t.__name__}]")
        try:
            t()
            print("  PASS")
        except _Skipped as e:
            skipped.append((t.__name__, str(e)))
            print(f"  SKIP {e}")
        except Exception as e:                                   # noqa: BLE001 - report, do not hide
            failed.append((t.__name__, repr(e)))
            print(f"  FAIL {e!r}")
    print("\n==== measured numbers ====")
    for k, v in NOTES:
        print(f"{k}: {v}")
    print(f"\n{len(tests) - len(failed) - len(skipped)}/{len(tests)} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped")
    for n, e in failed:
        print(f"FAILED {n}: {e}")
    for n, e in skipped:
        print(f"SKIPPED {n}: {e}")
    return 1 if failed else (2 if skipped else 0)


if __name__ == "__main__":
    raise SystemExit(main())
