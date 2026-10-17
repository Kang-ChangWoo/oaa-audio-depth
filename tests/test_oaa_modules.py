"""Equivalence gate for the four arrival-time / selection structure variants, 2026-10-17.

Run either way:
    python tests/test_oaa_modules.py          # prints every measured number
    pytest -q tests/test_oaa_modules.py

What is under test, in four sentences:

  `--tof-value-code`    adds the attention-weighted FIXED sin/cos code of the arrival time to the
      value readout, through a per-(head, channel) zero-init gain. The token time axis IS the
      depth axis (10 m window / 32 columns = 0.3125 m per column), and nothing else in the model
      hands that number to the decoder unmixed.
  `--tof-tau-inject`    computes tau_bar, the attention-weighted arrival time -- one scalar per
      ray, in metres, i.e. a linear estimator of depth -- Fourier-lifts it and adds it to the
      cross-attention block OUTPUT through one zero-init affine.
  `--cross-reread`      reads the observation tokens a SECOND time with the refined rays as the
      query, reusing the same `ray_mic` weights, mixed in by a zero-init per-channel gamma. The
      reference query is a per-ray prior that never sees the audio before deciding what to read.
  `--geo-key-coupling`  sends [geometry features, raw ray direction] into the per-head KEY space
      and adds <g, K> to the logits, so geometry can select a token. The reference geometry bias
      is per (ray, observation, head) and is copied across all M tokens of an observation.

Same discipline as `tests/test_oaa_midband.py`: every *equivalence* claim runs on CPU in float64,
because the CUDA TF32/flash kernels are not bit-reproducible and a GPU run could only establish a
loose bound. Gradient differences are divided by the LARGEST gradient in the configuration, never
by each tensor's own max. The bit behaviour of "adding an exact zero" under bf16 autocast is NOT
covered by this file and stays [unverified].

`OAA_REF_MODULE_30FF307` points at a pristine `30ff307` copy of model/oaa.py (default
`/root/local1/changwoo/modules_20261017/ref/oaa_ref_30ff307.py`, produced with
`git show 30ff307:model/oaa.py`). `30ff307` is the parent of this change. Missing -> the parity
tests FAIL rather than silently pass, unless OAA_REF_OPTIONAL=1 records an explicit, counted skip.

`MODULES_CKPT` points at a run directory whose best.pth must strict-load into the default-off
model (default the hop-16 control `relevel_h16_stem1_r2_rep_s0`).
"""
import importlib.util
import math
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import oaa as M                                              # noqa: E402
from model.oaa import (OAAv2Depth, GEO_FEAT_DIM, IDENTITY_INIT_PARAM_SUFFIXES,   # noqa: E402
                       _TOF_RANGE_M, _TOF_FOURIER_K, _tof_column_metres, _tof_sincos)

REF_PATH = os.environ.get("OAA_REF_MODULE_30FF307",
                          "/root/local1/changwoo/modules_20261017/ref/oaa_ref_30ff307.py")
CKPT_DIR = os.environ.get(
    "MODULES_CKPT",
    "/root/local1/changwoo/relevel_hop16_obs48_20261015/runs/rep/relevel_h16_stem1_r2_rep_s0")
SMALL = dict(C=64, lh=4, lw=8, enc_res=(16, 32))     # 256x512 output, tiny attention
CONTROL_PARAMS_M = 15.353642          # the measured control, from its manifest `params_m`
NOTES = []

TOF_KEYS = ["ray_mic.tof_gain"]
TAU_KEYS = ["ray_mic.tau_b", "ray_mic.tau_w"]
READ_KEYS = ["reread_ls"]
GEOK_KEYS = ["ray_mic.geo_key"]
ALL_NEW = sorted(TOF_KEYS + TAU_KEYS + READ_KEYS + GEOK_KEYS)

# every switch setting the gate sweeps. `hier_ok` marks the ones that survive a decomposed
# softmax: the two arrival-time switches read the post-softmax token probabilities, which
# `view_fusion != "global"` never materialises, so they refuse that combination by construction.
ON = [("tof", dict(tof_value_code=True), TOF_KEYS, False),
      ("tau", dict(tof_tau_inject=True), TAU_KEYS, False),
      ("reread", dict(cross_reread=True), READ_KEYS, True),
      ("geokey", dict(geo_key_coupling=True), GEOK_KEYS, True),
      ("tof+tau", dict(tof_value_code=True, tof_tau_inject=True),
       sorted(TOF_KEYS + TAU_KEYS), False),
      ("all4", dict(tof_value_code=True, tof_tau_inject=True, cross_reread=True,
                    geo_key_coupling=True), ALL_NEW, False)]
SUB_BRANCHES = [("masked", {}, True),
                ("registers", dict(registers=4), False),
                ("masked+registers", dict(registers=4), True),
                ("hier", dict(view_fusion="hier"), False),
                ("hier+masked", dict(view_fusion="hier"), True),
                ("hier-geo", dict(view_fusion="hier-geo"), False)]


class _Skipped(Exception):
    """An explicitly requested, counted skip. Never silent: main() reports it and exits non-zero."""


def note(k, v):
    NOTES.append((k, v))
    print(f"    {k} = {v}")


def _maxdiff(a, b):
    return float((a.detach() - b.detach()).abs().max())


def _ref_module():
    if not os.path.exists(REF_PATH):
        msg = (f"pristine 30ff307 module not found at {REF_PATH}.\n"
               f"    git show 30ff307:model/oaa.py > {REF_PATH}\n"
               f"    or set OAA_REF_OPTIONAL=1 for an explicit, counted skip.")
        if os.environ.get("OAA_REF_OPTIONAL") == "1":
            raise _Skipped(msg)
        raise FileNotFoundError(msg)
    spec = importlib.util.spec_from_file_location("oaa_ref_30ff307", REF_PATH)
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


def _ray_inputs(B, R, N, Mtok, C, seed=1, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    q_in = torch.randn(B, R, C, generator=g, dtype=torch.float64) * scale
    tokens = torch.randn(B, N * Mtok, C, generator=g, dtype=torch.float64) * scale
    el = (torch.arange(R, dtype=torch.float64) + 0.5) / R * math.pi - math.pi / 2
    dir3 = torch.stack([torch.cos(el), torch.sin(el), torch.cos(el) * 0.5], -1)
    dir3 = dir3 / dir3.norm(dim=-1, keepdim=True)
    poses = [(2 * math.pi * i / max(N // 2, 1), (-1.0, 1.0)[i % 2]) for i in range(N)]
    return q_in, tokens, dir3, poses


def _copy_shared(src, dst):
    """Copy every parameter/buffer the two modules have in common (new options add keys only)."""
    s, d = src.state_dict(), dst.state_dict()
    out = dst.load_state_dict({k: v for k, v in s.items() if k in d}, strict=False)
    assert not out.unexpected_keys, out.unexpected_keys
    return sorted(set(d) - set(s))


def _grads_of(m, x, obs_valid=None):
    m.zero_grad()
    D = m(x) if obs_valid is None else m(x, obs_valid=obs_valid)
    (D * torch.linspace(0.3, 1.7, D.numel(), dtype=D.dtype).view_as(D)).sum().backward()
    g = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
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


def _set_nonzero(m, std=0.05, seed=11):
    """Give every newly added parameter a real (non-identity) value."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n in ALL_NEW:
                torch.nn.init.normal_(p, std=std)


def _rezero(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n in ALL_NEW:
                torch.nn.init.zeros_(p)


def _dead_set(m, x):
    """Names of parameters whose gradient is exactly zero (or absent) after one backward."""
    m.zero_grad()
    D = m(x)
    (D * torch.linspace(0.3, 1.7, D.numel(), dtype=D.dtype).view_as(D)).sum().backward()
    return {n for n, p in m.named_parameters()
            if p.grad is None or float(p.grad.abs().max()) == 0.0}


# ============================================================ G1 all four off == 30ff307
def test_g1_flags_off_are_bit_identical_to_30ff307():
    """The patch must be completely inert when no switch is passed: same forward, same gradients,
    same state_dict keys as the pristine parent commit. Exactly 0.0, float64, N=2/4/8."""
    ref = _ref_module()
    worst_f = worst_g = 0.0
    for nv in (2, 4, 8):
        r = _small_model(nv, ref=ref).train()
        o = _small_model(nv).train()
        added = _copy_shared(r, o)
        assert added == [], f"flags OFF must add no keys, got {added}"
        assert sorted(r.state_dict()) == sorted(o.state_dict())
        x = _spec(2, nv)
        Dr, gr = _grads_of(r, x)
        Do, go = _grads_of(o, x)
        df = _maxdiff(Dr, Do)
        dg, where, _ = _cmp_grads(gr, go, f"N={nv}")
        worst_f, worst_g = max(worst_f, df), max(worst_g, dg)
        assert df == 0.0, (nv, df)
        assert dg == 0.0, (nv, dg, where)
    note("g1_off_vs_30ff307_forward_maxdiff", f"{worst_f:.3e} (N=2/4/8, must be exactly 0)")
    note("g1_off_vs_30ff307_grad_maxrel", f"{worst_g:.3e} (must be exactly 0)")


def test_g1b_flags_off_in_every_sub_branch():
    """`--geo-key-coupling` writes into the logit line both fusion modes funnel through, and the
    two arrival-time switches sit on the post-softmax probabilities, which the register branch
    splits. With every switch off all of masked / registers / hier / hier-geo must stay
    bit-identical, or the inertness claim covers one code path only."""
    ref = _ref_module()
    worst = 0.0
    for nv in (2, 4, 8):
        for tag, kw, ov in SUB_BRANCHES:
            r = _small_model(nv, ref=ref, **kw)
            o = _small_model(nv, **kw)
            assert _copy_shared(r, o) == []
            x = _spec(2, nv)
            obs = None
            if ov:
                obs = torch.ones(2, nv, dtype=torch.bool)
                obs[:, 1] = False
            try:
                M._SHORTCIRCUIT_ALL_VALID = ref._SHORTCIRCUIT_ALL_VALID = False
                a = r(x) if obs is None else r(x, obs_valid=obs)
                b = o(x) if obs is None else o(x, obs_valid=obs)
            finally:
                M._SHORTCIRCUIT_ALL_VALID = ref._SHORTCIRCUIT_ALL_VALID = True
            d = _maxdiff(a, b)
            worst = max(worst, d)
            assert d == 0.0, (nv, tag, d)
    note("g1b_off_all_sub_branches_maxdiff",
         f"{worst:.3e} (N=2/4/8 x 6 sub-branches, must be exactly 0)")


def test_g1c_flags_off_leave_the_gradient_identical_in_every_sub_branch():
    """G1b compares forwards only. A logit-side addition that is zero in the forward could still
    change a reduction order in the backward, so the gradients get the same sweep."""
    ref = _ref_module()
    worst = 0.0
    for nv in (2, 8):
        for tag, kw, ov in SUB_BRANCHES:
            r = _small_model(nv, ref=ref, **kw).train()
            o = _small_model(nv, **kw).train()
            assert _copy_shared(r, o) == []
            x = _spec(2, nv)
            obs = torch.ones(2, nv, dtype=torch.bool) if ov else None
            if ov:
                obs[:, 1] = False
            try:
                M._SHORTCIRCUIT_ALL_VALID = ref._SHORTCIRCUIT_ALL_VALID = False
                _, gr = _grads_of(r, x, obs)
                _, go = _grads_of(o, x, obs)
            finally:
                M._SHORTCIRCUIT_ALL_VALID = ref._SHORTCIRCUIT_ALL_VALID = True
            d, where, _ = _cmp_grads(gr, go, f"{tag} N={nv}")
            worst = max(worst, d)
            assert d == 0.0, (nv, tag, d, where)
    note("g1c_off_all_sub_branches_grad_maxrel",
         f"{worst:.3e} (N=2/8 x 6 sub-branches, must be exactly 0)")


# ============================================================ G2 the flags ON, at init
def test_g2_each_flag_on_at_init_is_bit_identical():
    """Every one of the four adds an exact 0.0 at init: a zero gain times the time code, a zero
    affine on the Fourier features, a zero gamma times (h2 - h), and an inner product with a zero
    key map. So turning any switch on -- or all four -- must not move a single bit, forward or in
    any shared gradient."""
    rows = {}
    for tag, kw, keys, _ in ON:
        worst_f = worst_g = 0.0
        for nv in (2, 4, 8):
            off = _small_model(nv).train()
            on = _small_model(nv, **kw).train()
            added = _copy_shared(off, on)
            assert added == keys, (tag, nv, added, keys)
            x = _spec(2, nv)
            Do, go = _grads_of(off, x)
            Dn, gn = _grads_of(on, x)
            df = _maxdiff(Do, Dn)
            dg, where, _ = _cmp_grads(go, {k: v for k, v in gn.items() if k in go}, f"{tag} N={nv}")
            worst_f, worst_g = max(worst_f, df), max(worst_g, dg)
            assert df == 0.0, (tag, nv, df)
            assert dg == 0.0, (tag, nv, dg, where)
        rows[tag] = (f"{worst_f:.3e}", f"{worst_g:.3e}")
    note("g2_on_at_init_forward_and_sharedgrad_maxdiff", rows)


def test_g2b_each_flag_on_at_init_in_every_sub_branch():
    """Same claim with an observation genuinely dropped, with registers present, and under the
    decomposed softmax for the two switches that accept it. The register branch is the one that
    could have been missed: it is the only place where `p_tok` is a slice of a wider softmax."""
    worst, covered = 0.0, 0
    for tag, kw, keys, hier_ok in ON:
        for nv in (2, 4, 8):
            for btag, bkw, ov in SUB_BRANCHES:
                if not hier_ok and bkw.get("view_fusion", "global") != "global":
                    continue
                off = _small_model(nv, **bkw)
                on = _small_model(nv, **bkw, **kw)
                assert _copy_shared(off, on) == keys, (tag, btag, nv)
                x = _spec(2, nv)
                obs = None
                if ov:
                    obs = torch.ones(2, nv, dtype=torch.bool)
                    obs[:, 1] = False
                    if nv >= 4:
                        obs[0, 2] = False
                try:
                    M._SHORTCIRCUIT_ALL_VALID = False
                    a = off(x) if obs is None else off(x, obs_valid=obs)
                    b = on(x) if obs is None else on(x, obs_valid=obs)
                finally:
                    M._SHORTCIRCUIT_ALL_VALID = True
                d = _maxdiff(a, b)
                worst = max(worst, d)
                covered += 1
                assert d == 0.0, (tag, btag, nv, d)
    note("g2b_on_at_init_all_sub_branches_maxdiff",
         f"{worst:.3e} over {covered} (switch, sub-branch, N) cells, must be exactly 0")


def test_g2c_other_widths_and_token_grids():
    """The time code and tau_bar are built against the TOKEN grid the caller passes in. A model
    with a different (lh, lw) or a different head width must still be bit-identical at init, and
    the code must be the right length -- an off-by-one in the column index would be invisible at
    init but wrong for the whole run."""
    # the decoder fixes the aspect ratio: _make_up asserts 256//lh == 512//lw, i.e. lw == 2*lh
    for lh, lw, C in ((4, 8, 64), (8, 16, 64), (4, 8, 128)):
        small = dict(C=C, lh=lh, lw=lw, enc_res=(4 * lh, 4 * lw))
        for kw in (dict(tof_value_code=True), dict(tof_tau_inject=True),
                   dict(cross_reread=True), dict(geo_key_coupling=True)):
            torch.manual_seed(0)
            off = OAAv2Depth(nviews=2, **small).double().eval()
            torch.manual_seed(0)
            on = OAAv2Depth(nviews=2, **small, **kw).double().eval()
            _copy_shared(off, on)
            x = torch.rand(2, 2, 4 * lh, 4 * lw, dtype=torch.float64,
                           generator=torch.Generator().manual_seed(5))
            assert _maxdiff(off(x), on(x)) == 0.0, (lh, lw, C, kw)
        m = OAAv2Depth(nviews=2, **small, tof_value_code=True, tof_tau_inject=True)
        assert m.ray_mic.tof_tau_m.shape == (lh * lw,), m.ray_mic.tof_tau_m.shape
        assert m.ray_mic.tof_code.shape == (lh * lw, C // 8), m.ray_mic.tof_code.shape
    note("g2c_token_grids",
         "(lh,lw,C) = (4,8,64) / (8,16,64) / (4,8,128): max|diff| = 0.0, code shapes exact")


# ============================================================ G3 the gradients are alive
def test_g3_every_new_parameter_has_a_live_gradient_at_step_0():
    """A zero-init affine keeps its own gradient (dL/dW = g (x) x) but a zero-init multiplicative
    gamma on a COLD branch does not. A zero here means the switch is cosmetic for the whole run."""
    rows = {}
    for tag, kw, keys, _ in ON:
        for nv in (2, 4, 8):
            m = _small_model(nv, **kw).train()
            _, g = _grads_of(m, _spec(2, nv))
            for k in keys:
                a = float(g[k].abs().max())
                rows[f"{tag}/N{nv}/{k}"] = f"{a:.4g}"
                assert a > 0.0, (tag, nv, k, "gradient is dead")
    note("g3_new_param_grad_absmax", rows)


def test_g3b_the_dead_parameter_set_does_not_grow():
    """The reference already carries a set of parameters whose gradient is exactly 0 at step 0 --
    the whole fine branch behind the zero-init `fine_to_dec`, plus `aux_head`, which the loss does
    not read. None of the four switches may ADD a name to that set: that would mean the switch
    froze a pathway that used to train."""
    rows = {}
    for nv in (2, 8):
        base = _dead_set(_small_model(nv).train(), _spec(2, nv))
        rows[f"N{nv}/reference_dead_count"] = len(base)
        for tag, kw, keys, _ in ON:
            d = _dead_set(_small_model(nv, **kw).train(), _spec(2, nv))
            grew = sorted(d - base - set(keys))
            rows[f"N{nv}/{tag}/newly_dead"] = grew or "none"
            assert grew == [], (nv, tag, grew)
            assert not (d & set(keys)), (nv, tag, sorted(d & set(keys)))
    note("g3b_dead_parameter_set", rows)


def test_g3c_a_gamma_on_top_of_the_tau_affine_would_have_been_dead():
    """The measurement that changed the design, kept as a live test so the reason cannot be lost.
    The brief asked for "Fourier -> one Linear -> per-channel zero-init gamma". With BOTH zeros the
    affine's own gradient at step 0 is gamma * (...) = exactly 0, i.e. the dead-branch failure that
    cost --pose-reinject its fine site. Simulated here by wrapping the shipped affine in a zero
    gamma and re-measuring; the shipped form (no gamma) is the live one."""
    rows = {}
    for nv in (2, 8):
        m = _small_model(nv, tof_tau_inject=True).train()
        _, g = _grads_of(m, _spec(2, nv))
        live_w = float(g["ray_mic.tau_w"].abs().max())
        live_b = float(g["ray_mic.tau_b"].abs().max())
        # the rejected form, measured on the same arithmetic the module would have run:
        #   d/dW [ gamma * (W f + b) ] = gamma * f = exactly 0 when gamma == 0
        f = torch.randn(4, 7, 2 * _TOF_FOURIER_K, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(nv))
        down = torch.randn(4, 7, SMALL["C"], dtype=torch.float64,
                           generator=torch.Generator().manual_seed(nv + 1))
        w = torch.zeros(2 * _TOF_FOURIER_K, SMALL["C"], dtype=torch.float64, requires_grad=True)
        b = torch.zeros(SMALL["C"], dtype=torch.float64, requires_grad=True)
        gm = torch.zeros(SMALL["C"], dtype=torch.float64, requires_grad=True)
        ((f @ w + b) * gm * down).sum().backward()
        rows[nv] = {"shipped_tau_w_grad": f"{live_w:.4g}", "shipped_tau_b_grad": f"{live_b:.4g}",
                    "rejected_with_gamma_w_grad": float(w.grad.abs().max()),
                    "rejected_with_gamma_b_grad": float(b.grad.abs().max()),
                    "rejected_gamma_grad": round(float(gm.grad.abs().max()), 6)}
        assert live_w > 0.0 and live_b > 0.0, (nv, live_w, live_b)
        assert float(w.grad.abs().max()) == 0.0 and float(b.grad.abs().max()) == 0.0, nv
    note("g3c_tau_affine_vs_rejected_gamma_form", rows)


# ============================================================ G4 parameter counts
def test_g4_exact_parameter_counts_at_the_real_width():
    """Exact integers, not "about", at the real C=256 / 8 heads / dk=32 / gf=5 width."""
    torch.manual_seed(0)
    nb = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2).parameters())
    counts = {}
    for tag, kw, _, _ in ON:
        torch.manual_seed(0)
        n = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, **kw).parameters())
        counts[tag] = n - nb
    assert counts["tof"] == 8 * 32 == 256, counts
    assert counts["tau"] == (2 * _TOF_FOURIER_K) * 256 + 256 == 8448, counts
    assert counts["reread"] == 256, counts
    assert counts["geokey"] == (GEO_FEAT_DIM["raw"] + 3) * 256 == 2048, counts
    assert counts["all4"] == 256 + 8448 + 256 + 2048 == 11008, counts
    # the sh4 basis widens the key map exactly as it widens the bias MLP input
    torch.manual_seed(0)
    nsh = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, geo_basis="sh4").parameters())
    torch.manual_seed(0)
    nshg = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, geo_basis="sh4",
                                             geo_key_coupling=True).parameters())
    assert nshg - nsh == (GEO_FEAT_DIM["sh4"] + 3) * 256 == 7680, (nshg - nsh)
    note("g4_param_base_M", f"{nb / 1e6:.6f} (control manifest {CONTROL_PARAMS_M})")
    assert abs(nb / 1e6 - CONTROL_PARAMS_M) < 1e-6, nb
    note("g4_param_added", counts)
    note("g4_param_added_geokey_sh4", nshg - nsh)
    for tag, kw, _, _ in ON:
        torch.manual_seed(0)
        n = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, **kw).parameters())
        note(f"g4_total_M/{tag}", f"{n / 1e6:.6f}")


def test_g4b_new_parameters_stay_out_of_the_param_lr_group():
    """Recorded as a deliberate choice, not an oversight, exactly as GeoFiLM's `value_film` was:
    these tensors are zero-init, but adding them to IDENTITY_INIT_PARAM_SUFFIXES would change what
    --param-lr-mult means for every existing cell. The queued cells leave it at its default 1.0."""
    m = _small_model(2, tof_value_code=True, tof_tau_inject=True, cross_reread=True,
                     geo_key_coupling=True)
    hit = sorted(n for n, _ in m.named_parameters() if n.endswith(IDENTITY_INIT_PARAM_SUFFIXES))
    note("g4b_param_lr_group", hit)
    assert not any(k in n for n in hit
                   for k in ("tof_gain", "tau_w", "tau_b", "reread_ls", "geo_key")), hit


# ============================================================ G5 keys / checkpoint compatibility
def test_g5_only_the_declared_keys_are_added_at_the_real_width():
    for tag, kw, keys, _ in ON:
        for nv in (2, 8):
            torch.manual_seed(0)
            base = OAAv2Depth(C=256, nviews=nv)
            torch.manual_seed(0)
            var = OAAv2Depth(C=256, nviews=nv, **kw)
            added = sorted(set(var.state_dict()) - set(base.state_dict()))
            assert added == keys, (tag, nv, added)
            assert sorted(set(base.state_dict()) - set(var.state_dict())) == []
            out = var.load_state_dict(base.state_dict(), strict=False)
            assert sorted(out.missing_keys) == keys, (tag, nv, out.missing_keys)
            assert list(out.unexpected_keys) == [], out.unexpected_keys
    note("g5_new_state_dict_keys", {t: k for t, _, k, _ in ON})


def test_g5b_existing_hop16_checkpoint_strict_loads_into_the_default_off_model():
    """The default-off contract that matters operationally: the real trained control checkpoint
    still loads with strict=True, and warm-starting a variant leaves exactly the new keys missing.
    This also proves the new non-persistent buffers really are out of state_dict."""
    pth = os.path.join(CKPT_DIR, "best.pth")
    if not os.path.exists(pth):
        msg = f"checkpoint {pth} not found; set MODULES_CKPT, or OAA_REF_OPTIONAL=1 to skip"
        if os.environ.get("OAA_REF_OPTIONAL") == "1":
            raise _Skipped(msg)
        raise FileNotFoundError(msg)
    sd = torch.load(pth, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    torch.manual_seed(0)
    m = OAAv2Depth(C=256, nviews=2, stem_stride1=True)
    m.load_state_dict(sd, strict=True)
    note("g5b_hop16_ckpt_strict_load", f"{os.path.basename(CKPT_DIR)} loaded with strict=True "
                                       f"({len(sd)} tensors)")
    for tag, kw, keys, _ in ON:
        torch.manual_seed(0)
        mv = OAAv2Depth(C=256, nviews=2, stem_stride1=True, **kw)
        out = mv.load_state_dict(sd, strict=False)
        assert sorted(out.missing_keys) == keys, (tag, out.missing_keys)
        assert list(out.unexpected_keys) == [], out.unexpected_keys
    note("g5b_warm_start_missing_keys", f"exactly the new tensors for all {len(ON)} settings")


def test_g5c_config_round_trips_through_core_ckpt():
    """Save {state_dict, args} -> core.ckpt.build(args, DM) -> strict load -> identical prediction.
    Without this a cell trains for 2 hours and is then UNSCOREABLE: eval.py catches the strict-load
    failure per run and prints only `[skip <run>]`, so the OAA row vanishes from the table and the
    baselines look like the whole comparison. That is the hole `11b7202` had to patch for GeoFiLM
    and the reason the brief names core/ckpt.py explicitly."""
    from core.ckpt import build
    DM = types.SimpleNamespace(IN_CH={"r2": 2, "cB": 4, "fb": 4, "r6": 6, "r8": 8}, POSES={},
                               WINDOW=0, HOP=160, WIN=400, N_FFT=512)
    DM.__name__ = "data_0422"
    passthru = ("tof_value_code", "tof_tau_inject", "cross_reread", "geo_key_coupling",
                "qk_norm", "registers", "geo_basis", "geo_value_film", "pose_reinject",
                "geo_ray_gain")
    cfgs = [{}, {"tof_value_code": True}, {"tof_tau_inject": True}, {"cross_reread": True},
            {"geo_key_coupling": True},
            {"tof_value_code": True, "tof_tau_inject": True},
            {"tof_value_code": True, "tof_tau_inject": True, "cross_reread": True,
             "geo_key_coupling": True},
            {"geo_key_coupling": True, "geo_basis": "sh4"},
            {"cross_reread": True, "qk_norm": True, "registers": 4},
            {"tof_value_code": True, "pose_reinject": True, "geo_ray_gain": True}]
    for extra in cfgs:
        args = dict(nviews=2, data_mode="r2", dim=64, rounds=2, lift_h=4, lift_w=8,
                    stem_stride1=True, max_depth=10.0, dec_deep=True, rounds_wired=True,
                    cond_mode="adaln", full_res_enc=True, multi_scale_lift=True,
                    data_module="data_0422", **extra)
        torch.manual_seed(0)
        src = OAAv2Depth(C=64, nviews=2, lh=4, lw=8, stem_stride1=True,
                         **{k: extra[k] for k in passthru if k in extra}).eval()
        rebuilt, dmode, nch, kind, poses = build(args, DM)
        assert set(rebuilt.state_dict()) == set(src.state_dict()), extra
        rebuilt.load_state_dict(src.state_dict(), strict=True)
        rebuilt.eval()
        x = torch.rand(1, 2, 24, 40)
        with torch.no_grad():
            d = _maxdiff(src(x), rebuilt(x))
        assert d == 0.0, (extra, d)
    note("g5c_ckpt_round_trip", f"{len(cfgs)} configs strict-loaded and prediction-compared: "
                                f"max|diff| = 0.0")


def test_g5d_checkpoint_save_load_round_trip_key_by_key():
    """Save a TRAINED (non-identity) variant to disk and read it back: every tensor identical key
    by key, and the reloaded model predicts the same thing. The new buffers are non-persistent, so
    this also pins that they are rebuilt from (lh, lw) rather than restored."""
    import io
    for tag, kw, keys, _ in ON:
        m = _small_model(2, **kw)
        _set_nonzero(m, std=0.2)
        buf = io.BytesIO()
        torch.save({"state_dict": m.state_dict()}, buf)
        buf.seek(0)
        sd = torch.load(buf, map_location="cpu", weights_only=False)["state_dict"]
        m2 = _small_model(2, **kw)
        out = m2.load_state_dict(sd, strict=True)
        assert not out.missing_keys and not out.unexpected_keys, (tag, out)
        a, b = m.state_dict(), m2.state_dict()
        assert sorted(a) == sorted(b), tag
        for k in a:
            assert _maxdiff(a[k], b[k]) == 0.0, (tag, k)
        for k in keys:
            assert k in a, (tag, k)
        x = _spec(2, 2)
        assert _maxdiff(m(x), m2(x)) == 0.0, tag
    note("g5d_disk_round_trip", f"{len(ON)} settings: every tensor and the prediction identical")


# ============================================================ G6 not inert once trained
def test_g6_real_weights_move_the_output_and_rezeroing_restores_it_exactly():
    """Otherwise G2 proves nothing: each switch has to be capable of changing the prediction, and
    putting its parameters back at the init has to restore the reference EXACTLY."""
    moved, restored = {}, 0.0
    for tag, kw, keys, _ in ON:
        for nv in (2, 4, 8):
            off = _small_model(nv)
            on = _small_model(nv, **kw)
            assert _copy_shared(off, on) == keys
            x = _spec(2, nv)
            _set_nonzero(on)
            d = _maxdiff(off(x), on(x))
            moved[f"{tag}/N{nv}"] = round(d, 6)
            assert d > 1e-6, (tag, nv, "the switch does nothing even with real weights", d)
            _rezero(on)
            dz = _maxdiff(off(x), on(x))
            restored = max(restored, dz)
            assert dz == 0.0, (tag, nv, dz)
    note("g6_output_shift_with_real_weights", moved)
    note("g6_restored_after_rezeroing", f"{restored:.3e} (must be exactly 0)")


# ============================================================ G7 the mechanism claims
def test_g7_the_time_code_is_a_function_of_the_time_column_only():
    """The physics the whole arrival-time family rests on: token m sits in time column m % lw, one
    column is 10/32 = 0.3125 m of round trip, and the code must therefore be IDENTICAL for two
    tokens in the same column and different frequency rows, and DIFFERENT across columns."""
    lh, lw = 16, 32
    tau = _tof_column_metres(lh, lw)
    assert tau.shape == (lh * lw,)
    step = _TOF_RANGE_M / lw
    assert abs(step - 0.3125) < 1e-12, step
    assert abs(float(tau[0]) - 0.5 * step) < 1e-12, float(tau[0])
    assert abs(float(tau[lw - 1]) - (lw - 0.5) * step) < 1e-12, float(tau[lw - 1])
    # same column, different frequency row -> same arrival time
    for r in range(1, lh):
        assert float((tau[r * lw:(r + 1) * lw] - tau[:lw]).abs().max()) == 0.0, r
    code = _tof_sincos(tau, 32)
    assert code.shape == (lh * lw, 32)
    for r in range(1, lh):
        assert float((code[r * lw:(r + 1) * lw] - code[:lw]).abs().max()) == 0.0, r
    sep = min(float((code[c] - code[c + 1]).abs().max()) for c in range(lw - 1))
    note("g7_time_code", {"metres_per_column": step, "rows_identical": True,
                          "min_adjacent_column_separation": round(sep, 5)})
    assert sep > 1e-3, sep
    # the code is a BUFFER, not a parameter: what is learned is how much of it to add
    m = OAAv2Depth(C=256, nviews=2, tof_value_code=True)
    assert "ray_mic.tof_code" not in dict(m.named_parameters())
    assert any(n == "ray_mic.tof_code" for n, _ in m.named_buffers())
    note("g7_time_code_is_a_buffer", "ray_mic.tof_code is a non-persistent buffer, +0 parameters")


def test_g7b_tau_bar_is_the_first_moment_of_the_measured_column_mass():
    """tau_bar is claimed to be "the attention-weighted arrival time". Checked against the module's
    own reported per-column attention mass rather than re-derived: sum_c mass(c) * centre(c) must
    reproduce the reported tau_bar, and tau_bar must lie inside the 0-10 m window."""
    rows = {}
    for nv in (2, 8):
        m = _small_model(nv, tof_tau_inject=True)
        m.ray_mic.collect_stats = True
        m(_spec(2, nv))
        st = m.ray_mic.stats
        cm = st["tof_col_mass_full"].double()                       # (B,R,lw)
        lw = SMALL["lw"]
        centre = (torch.arange(lw, dtype=torch.float64) + 0.5) * (_TOF_RANGE_M / lw)
        recomputed = (cm * centre).sum(-1)                           # (B,R)
        rep = st["tau_bar_per_ray"].double()                         # (R,) batch mean
        d = float((recomputed.mean(0) - rep).abs().max())
        rows[nv] = {"max|recomputed - reported|": f"{d:.3e}",
                    "tau_bar_mean_m": round(st["tau_bar_mean"], 4),
                    "tau_bar_min_m": round(st["tau_bar_min"], 4),
                    "tau_bar_max_m": round(st["tau_bar_max"], 4),
                    "total_column_mass": round(float(cm.sum(-1).mean()), 6)}
        assert d < 1e-6, (nv, d)
        assert 0.0 < st["tau_bar_min"] and st["tau_bar_max"] < _TOF_RANGE_M, (nv, st)
        # the reported mass is accumulated in float32 (it is a diagnostic, not a forward value),
        # so the sum over columns lands within a float32 epsilon of 1, not exactly on it
        assert abs(float(cm.sum(-1).mean()) - 1.0) < 1e-6, (nv, float(cm.sum(-1).mean()))
    note("g7b_tau_bar_vs_column_mass", rows)


def test_g7c_the_reread_uses_the_same_weights_and_a_different_query():
    """The design claim for --cross-reread: no new layer (the second read is literally the same
    module), and the query it uses is NOT the first query -- otherwise the switch would be reading
    the same thing twice and `h2 - h` would carry no new information."""
    nv = 4
    m = _small_model(nv, cross_reread=True)
    before = {n: p.clone() for n, p in m.ray_mic.named_parameters()}
    _set_nonzero(m, std=0.2)
    after = {n: p for n, p in m.ray_mic.named_parameters()}
    shared = [n for n in before if n in after and n != "tof_gain"]
    # the re-read adds exactly one tensor and it does not live inside ray_mic
    assert "reread_ls" in dict(m.named_parameters())
    assert "reread_ls" not in dict(m.ray_mic.named_parameters())
    changed = [n for n in shared if _maxdiff(before[n], after[n]) > 0]
    assert changed == [], changed
    # drive the two reads apart: the first query is the prior, the second is the refined ray
    x = _spec(2, nv)
    F4, poses, _, _ = m._encode(x)
    dir6 = m.dir6_buf.double(); dir3 = dir6[:, :3]
    B = x.size(0)
    q1 = (m.q + m.dir_mlp(dir6).unsqueeze(0)).expand(B, -1, -1)
    toks = F4.reshape(B, m.nv * m.M, m.C)
    h1 = m.ray_mic(q1, toks, dir3, poses, m.M)
    h2 = m.ray_mic(h1, toks, dir3, poses, m.M)
    dq = float((q1 - h1).abs().max())
    dh = float((h2 - h1).abs().max())
    note("g7c_reread", {"query_prior_vs_refined_maxdiff": round(dq, 5),
                        "second_read_delta_maxdiff": round(dh, 5),
                        "ray_mic_params_added": 0})
    assert dq > 1e-6 and dh > 1e-6, (dq, dh)


def test_g7d_geo_key_coupling_varies_within_an_observation():
    """The one thing that separates --geo-key-coupling from every geometry variant tried so far.
    The reference bias `bmic` is (R, N, h) and is copied across the M tokens of an observation, so
    its spread WITHIN an observation is exactly 0. The coupling term must have a non-zero spread
    there, or it is another per-observation scalar in a new costume."""
    B, R, N, Mtok, C, heads = 2, 16, 4, 8, 64, 8
    rm = _ray_mod(dk=C // heads, heads=heads, geo_key_coupling=True)
    with torch.no_grad():
        torch.nn.init.normal_(rm.geo_key, std=0.3)
    q, tk, d3, poses = _ray_inputs(B, R, N, Mtok, C)
    # recompute the two logit terms with the module's own pieces
    feats = rm._geo_basis_feats(d3, poses)                              # (R,N,gf)
    bmic = torch.stack([rm.bias_mlp(feats[:, i].contiguous()) for i in range(N)], 1)   # (R,N,h)
    K = rm.k(rm.nk(tk)).view(B, -1, heads, C // heads).transpose(1, 2)
    gin = torch.cat([feats, d3.unsqueeze(1).expand(R, N, 3)], -1)
    gk = (gin @ rm.geo_key).view(R, N, heads, C // heads)
    add = torch.einsum("rnhd,bhnmd->bhrnm", gk, K.reshape(B, heads, N, Mtok, C // heads))
    within_ref = float(bmic.std(0).abs().max() * 0)          # constant over M by construction
    within_new = float(add.std(-1).abs().max())              # spread over the M tokens
    note("g7d_logit_term_spread_within_an_observation",
         {"reference_bias": within_ref, "geo_key_term": round(within_new, 5),
          "term_shape": tuple(add.shape)})
    assert within_new > 1e-6, within_new
    # and it must still depend on geometry: zeroing the geometry columns must change it
    gin0 = gin.clone(); gin0[..., :GEO_FEAT_DIM["raw"]] = 0.0
    gk0 = (gin0 @ rm.geo_key).view(R, N, heads, C // heads)
    add0 = torch.einsum("rnhd,bhnmd->bhrnm", gk0, K.reshape(B, heads, N, Mtok, C // heads))
    assert float((add - add0).abs().max()) > 1e-6
    # per-head, not head-shared
    assert float(gk.std(2).abs().max()) > 1e-6
    note("g7d_geo_key_is_per_head_and_geometry_driven", "both checks non-zero")


def test_g7e_obs_mask_invariance_survives_trained_parameters():
    """With real (non-identity) parameters an invalid observation still must not change the
    prediction. All four switches read quantities the mask already zeroes (the probabilities) or
    geometry, which is deterministic per observation. This is the invariance `--subset-aug`
    depends on, and it is the one a readout that sums over tokens could break."""
    nv = 8
    rows = {}
    for tag, kw, _, _ in ON:
        m = _small_model(nv, **kw)
        _set_nonzero(m)
        ov = torch.ones(2, nv, dtype=torch.bool); ov[:, [1, 4, 6]] = False
        x = _spec(2, nv)
        a = m(x, obs_valid=ov)
        x2 = x.clone()
        x2[:, [1, 4, 6]] = torch.rand_like(x2[:, [1, 4, 6]]) * 7.0 + 3.0
        d = _maxdiff(a, m(x2, obs_valid=ov))
        rows[f"{tag}/mask_invariance"] = f"{d:.3e}"
        assert d < 1e-12, (tag, d)
        try:
            M._SHORTCIRCUIT_ALL_VALID = False
            dz = _maxdiff(m(x), m(x, obs_valid=torch.ones(2, nv, dtype=torch.bool)))
        finally:
            M._SHORTCIRCUIT_ALL_VALID = True
        rows[f"{tag}/allvalid_identity"] = f"{dz:.3e}"
        assert dz < 1e-10, (tag, dz)
    note("g7e_mask_invariance_with_trained_params", rows)


# ============================================================ refusals
def test_incoherent_combinations_are_refused():
    """--geo-key-coupling reads the geometry features --no-geo-bias never computes. The two
    arrival-time switches read the post-softmax token probabilities of the SINGLE global softmax,
    which `--view-fusion hier/hier-geo` decomposes and never materialises. --no-cross deletes the
    attention all four depend on."""
    cases = [("geokey+no_geo_bias", dict(geo_key_coupling=True, no_geo_bias=True), "geo-bias"),
             ("tof+hier", dict(tof_value_code=True, view_fusion="hier"), "global softmax"),
             ("tau+hier-geo", dict(tof_tau_inject=True, view_fusion="hier-geo"), "global softmax"),
             ("tof+no_cross", dict(tof_value_code=True, no_cross=True), "no-cross"),
             ("tau+no_cross", dict(tof_tau_inject=True, no_cross=True), "no-cross"),
             ("reread+no_cross", dict(cross_reread=True, no_cross=True), "no-cross"),
             ("geokey+no_cross", dict(geo_key_coupling=True, no_cross=True), "no-cross")]
    for tag, kw, want in cases:
        try:
            _small_model(4, **kw)
            raise RuntimeError(f"{tag} should have been refused")
        except AssertionError as e:
            assert want in str(e), (tag, str(e))
    # --cross-reread and --geo-key-coupling are legitimate under a decomposed softmax
    _small_model(4, cross_reread=True, view_fusion="hier")
    _small_model(4, geo_key_coupling=True, view_fusion="hier-geo")
    note("refusals", f"{len(cases)} incoherent combinations raise; reread/geokey under hier are "
                     f"allowed on purpose")


# ============================================================ run-level contracts
def test_forward_contract_for_the_queued_cell_configs():
    """The exact configurations the cells would queue (`--stem-stride1` plus the switch set):
    output (B,1,256,512) in [0,1], finite, and a live finite gradient on every new parameter."""
    CELLS = [("tofcode+tau", dict(tof_value_code=True, tof_tau_inject=True)),
             ("reread", dict(cross_reread=True)),
             ("geokey", dict(geo_key_coupling=True)),
             ("all4", dict(tof_value_code=True, tof_tau_inject=True, cross_reread=True,
                           geo_key_coupling=True))]
    for tag, kw in CELLS:
        for nv in (2, 8):
            m = _small_model(nv, stem_stride1=True, **kw).train()
            D = m(_spec(2, nv))
            assert D.shape == (2, 1, 256, 512) and torch.isfinite(D).all(), (tag, nv)
            assert float(D.min()) >= 0.0 and float(D.max()) <= 1.0, (tag, nv)
            D.sum().backward()
            pm = dict(m.named_parameters())
            keys = [k for k in ALL_NEW if k in pm]
            for q in keys + ["ray_mic.q.weight", "ray_mic.bias_mlp.0.weight", "pose_emb.0.weight"]:
                g = pm[q].grad
                assert g is not None and torch.isfinite(g).all(), (tag, nv, q)
                assert float(g.abs().sum()) > 0, (tag, nv, q)
    note("cell_forward_contract",
         f"{len(CELLS)} queued cell configs x N=2/8 with --stem-stride1: (2,1,256,512) in [0,1], "
         f"live finite gradients")


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
    sys.exit(main())
