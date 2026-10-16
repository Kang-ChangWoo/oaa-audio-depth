"""Equivalence gate for the C1 content abstention gate (`--film-lse`).

Run either way:
    python tests/test_oaa_film_lse.py          # prints every measured number
    pytest -q tests/test_oaa_film_lse.py

What is under test, in one sentence: the GeoFiLM multiplier gets ONE extra input column, the
per-(ray, observation, head) `logsumexp` of the pre-softmax logits over that observation's M
tokens, so the film can read *how much there is to hear* and not only *where the geometry points*.
Driven by a single zero-init weight vector of shape (heads,) -- an affine first layer, so the
multiplier is exactly 1.0 at init AND the weight has a live gradient. +8 parameters.

Same discipline as `tests/test_oaa_geofilm.py`: every *equivalence* claim runs on CPU in float64,
because the CUDA TF32/flash kernels are not bit-reproducible and a GPU run could only establish a
loose bound. Gradient differences are divided by the LARGEST gradient in the configuration, never
by each tensor's own max.

`OAA_REF_MODULE_BD69F42` points at a pristine `bd69f42` copy of model/oaa.py (default
`/root/local1/changwoo/filmlse_20261016/ref/oaa_ref_bd69f42.py`, produced with
`git show bd69f42:model/oaa.py`). `bd69f42` is the parent of this change -- GeoFiLM `bb4aa82` plus
the scoring fix `11b7202` plus SwiGLU `bd69f42` -- so "the flag off reproduces bd69f42 bit for
bit" is the claim. Missing -> the parity tests FAIL rather than silently pass, unless
OAA_REF_OPTIONAL=1 records an explicit, counted skip.

`ARCHFIX_CKPT` points at a run directory whose best.pth must strict-load into the default-off
model (default the hop-16 control `relevel_h16_stem1_r2_rep_s0`).

Two negative results from the literature review are encoded here as tests, not as prose:
  * a 2-layer zero-init head cold-starts its inner layer -- this gate is ONE layer (G8);
  * the exponential null-token form cannot have bit-identity and a live gradient at the same time,
    so this gate is affine (G4 measures that both hold here).
"""
import importlib.util
import math
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import oaa as M                                              # noqa: E402
from model.oaa import (OAAv2Depth, RayMicAttn, GEO_FEAT_DIM,            # noqa: E402
                       IDENTITY_INIT_PARAM_SUFFIXES)

REF_PATH = os.environ.get("OAA_REF_MODULE_BD69F42",
                          "/root/local1/changwoo/filmlse_20261016/ref/oaa_ref_bd69f42.py")
CKPT_DIR = os.environ.get(
    "ARCHFIX_CKPT",
    "/root/local1/changwoo/relevel_hop16_obs48_20261015/runs/rep/relevel_h16_stem1_r2_rep_s0")
SMALL = dict(C=64, lh=4, lw=8, enc_res=(16, 32))     # 256x512 output, tiny attention
PARAM_CEILING_M = 15.45                              # PREREG hard ceiling; the paper's #P prints 15.4
FILM_KEYS = ["ray_mic.value_film.bias", "ray_mic.value_film.weight"]
LSE_KEYS = ["ray_mic.film_lse_w"]
BOTH_KEYS = sorted(FILM_KEYS + LSE_KEYS)
NOTES = []


class _Skipped(Exception):
    """An explicitly requested, counted skip. Never silent: main() reports it and exits non-zero."""


def note(k, v):
    NOTES.append((k, v))
    print(f"    {k} = {v}")


def _maxdiff(a, b):
    return float((a.detach() - b.detach()).abs().max())


def _ref_module():
    if not os.path.exists(REF_PATH):
        msg = (f"pristine bd69f42 module not found at {REF_PATH}.\n"
               f"    git show bd69f42:model/oaa.py > {REF_PATH}\n"
               f"    or set OAA_REF_OPTIONAL=1 for an explicit, counted skip.")
        if os.environ.get("OAA_REF_OPTIONAL") == "1":
            raise _Skipped(msg)
        raise FileNotFoundError(msg)
    spec = importlib.util.spec_from_file_location("oaa_ref_bd69f42", REF_PATH)
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


# ==================================================== L1/L2 the flag off == bd69f42
def test_l1_flag_off_is_bit_identical_to_bd69f42():
    """The patch must be completely inert when `--film-lse` is not passed: same forward, same
    gradients, same state_dict keys as the pristine parent. Exactly 0.0, float64, N = 2/4/8.
    Run with the film OFF and with the film ON, because the patch edits the GeoFiLM block too."""
    ref = _ref_module()
    worst_f = worst_g = 0.0
    for nv in (2, 4, 8):
        for kw in ({}, dict(geo_value_film=True)):
            r = _small_model(nv, ref=ref, **kw).train()
            o = _small_model(nv, **kw).train()
            added = _copy_shared(r, o)
            assert added == [], f"flag OFF must add no keys, got {added}"
            assert sorted(r.state_dict()) == sorted(o.state_dict())
            x = _spec(2, nv)
            Dr, gr = _grads_of(r, x)
            Do, go = _grads_of(o, x)
            df = _maxdiff(Dr, Do)
            dg, where, _ = _cmp_grads(gr, go, f"N={nv} {kw}")
            worst_f, worst_g = max(worst_f, df), max(worst_g, dg)
            assert df == 0.0, (nv, kw, df)
            assert dg == 0.0, (nv, kw, dg, where)
    note("l1_off_vs_bd69f42_forward_maxdiff",
         f"{worst_f:.3e} (N=2/4/8 x film off/on, must be exactly 0)")
    note("l2_off_vs_bd69f42_grad_maxrel", f"{worst_g:.3e} (must be exactly 0)")


def test_l1b_flag_off_masked_and_register_branches_match_bd69f42():
    """The lse column is read inside all THREE global-softmax sub-branches. Each one has to stay
    bit-identical with the flag off, or the inertness claim covers only one code path."""
    ref = _ref_module()
    worst = 0.0
    for nv in (2, 4, 8):
        for tag, kw, ov in (("masked", dict(geo_value_film=True), True),
                            ("registers", dict(geo_value_film=True, registers=4), False),
                            ("masked+registers", dict(geo_value_film=True, registers=4), True)):
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
    note("l1b_off_all_three_branches_maxdiff",
         f"{worst:.3e} (9 configurations x 3 sub-branches, must be exactly 0)")


# ==================================================== L3 the flag ON, at init
def test_l3_lse_on_at_init_is_bit_identical():
    """film = 1 + Linear(feats) + w * lse with w zero means the added term is exactly 0.0, and
    `film + 0.0 == film` in IEEE-754, so turning the switch ON must not move a single bit at init.
    This is the whole reason the column is affine rather than an exponential null token."""
    worst_f = worst_g = 0.0
    for nv in (2, 4, 8):
        off = _small_model(nv, geo_value_film=True).train()
        on = _small_model(nv, geo_value_film=True, film_lse=True).train()
        added = _copy_shared(off, on)
        assert added == LSE_KEYS, added
        x = _spec(2, nv)
        Do, go = _grads_of(off, x)
        Dn, gn = _grads_of(on, x)
        df = _maxdiff(Do, Dn)
        dg, where, _ = _cmp_grads(go, {k: v for k, v in gn.items() if k in go}, f"N={nv}")
        worst_f, worst_g = max(worst_f, df), max(worst_g, dg)
        assert df == 0.0, (nv, df)
        assert dg == 0.0, (nv, dg, where)
    note("l3_lse_on_at_init_forward_maxdiff", f"{worst_f:.3e} (N=2/4/8, must be exactly 0)")
    note("l3_lse_on_at_init_shared_grad_maxrel", f"{worst_g:.3e} (must be exactly 0)")


def test_l3b_lse_on_at_init_in_the_masked_and_register_branches():
    """A dropped observation makes its whole logit row -inf, so logsumexp is -inf and 0.0 * -inf
    would be nan. This is the test that would catch that: masked, registers, and both."""
    worst = 0.0
    for nv in (2, 4, 8):
        for tag, kw in (("masked", {}), ("masked+registers", dict(registers=4))):
            off = _small_model(nv, geo_value_film=True, **kw)
            on = _small_model(nv, geo_value_film=True, film_lse=True, **kw)
            assert _copy_shared(off, on) == LSE_KEYS
            x = _spec(2, nv)
            obs = torch.ones(2, nv, dtype=torch.bool)
            obs[:, 1] = False
            if nv >= 4:
                obs[0, 2] = False
            a, b = off(x, obs_valid=obs), on(x, obs_valid=obs)
            assert torch.isfinite(b).all(), (nv, tag, "nan/inf leaked out of the lse column")
            d = _maxdiff(a, b)
            worst = max(worst, d)
            assert d == 0.0, (nv, tag, d)
        off = _small_model(nv, geo_value_film=True, registers=4)
        on = _small_model(nv, geo_value_film=True, film_lse=True, registers=4)
        assert _copy_shared(off, on) == LSE_KEYS
        x = _spec(2, nv)
        d = _maxdiff(off(x), on(x))
        worst = max(worst, d)
        assert d == 0.0, (nv, "registers", d)
    note("l3b_lse_on_masked_register_branches_maxdiff",
         f"{worst:.3e} (N=2/4/8 x masked / masked+registers / registers, must be exactly 0)")


def test_l3c_a_fully_dropped_observation_is_finite_with_a_TRAINED_weight():
    """The nan path only bites once the weight is non-zero: 0.0 * -inf is nan, w * -inf is -inf,
    and both must be neutralised. Drop EVERY observation but one, give the column a real weight,
    and require a finite prediction plus finite gradients."""
    nv = 8
    m = _small_model(nv, geo_value_film=True, film_lse=True).train()
    with torch.no_grad():
        torch.nn.init.normal_(m.ray_mic.film_lse_w, std=0.05)
        torch.nn.init.normal_(m.ray_mic.value_film.weight, std=0.05)
    ov = torch.ones(2, nv, dtype=torch.bool)
    ov[:, 1:] = False                                  # exactly one observation survives
    D = m(_spec(2, nv), obs_valid=ov)
    assert torch.isfinite(D).all(), "a fully dropped observation produced nan/inf"
    D.sum().backward()
    g = m.ray_mic.film_lse_w.grad
    assert g is not None and torch.isfinite(g).all(), "lse gradient is nan with dropped views"
    note("l3c_one_surviving_observation", f"finite prediction, finite grad absmax "
                                         f"{float(g.abs().max()):.4f}")


def test_l3d_reference_state_dict_loads_with_only_the_new_keys_missing():
    """An existing checkpoint has to load into the gate-on model: missing exactly the new tensors,
    nothing unexpected. That is what makes the switch deployable on a warm start."""
    for nv in (2, 8):
        base = _small_model(nv)
        on = _small_model(nv, geo_value_film=True, film_lse=True)
        out = on.load_state_dict(base.state_dict(), strict=False)
        assert sorted(out.missing_keys) == BOTH_KEYS, out.missing_keys
        assert list(out.unexpected_keys) == [], out.unexpected_keys
    note("l3d_loadstate_missing", BOTH_KEYS)
    note("l3d_loadstate_unexpected", [])


# ==================================================== L4 the gradient is alive
def test_l4_film_lse_w_gets_a_gradient_at_step_0():
    """The pair of properties the literature review could not get from the canonical null-token
    form: the multiplier is exactly 1.0 AND dL/dw != 0 at init. If this is 0 the switch is
    cosmetic and the cell is a re-run of GeoFiLM."""
    rows = {}
    for nv in (2, 4, 8):
        m = _small_model(nv, geo_value_film=True, film_lse=True).train()
        _, g = _grads_of(m, _spec(2, nv))
        gw = float(g["ray_mic.film_lse_w"].abs().max())
        rows[nv] = round(gw, 4)
        assert gw > 0.0, (nv, "film_lse_w gradient is dead")
        # and the geometry half is still alive, i.e. the two columns do not cancel each other
        assert float(g["ray_mic.value_film.weight"].abs().max()) > 0.0, nv
        assert float(g["ray_mic.value_film.bias"].abs().max()) > 0.0, nv
    note("l4_film_lse_w_grad_absmax", rows)


def test_l4b_the_gradient_is_not_an_artefact_of_a_constant_lse():
    """If logsumexp were constant across observations the column would be a per-head bias, i.e. a
    duplicate of value_film.bias. Measure the spread of the driving signal itself."""
    rm = _ray_mod(geo_value_film=True, film_lse=True)
    q, tk, d3, poses = _ray_inputs(2, 16, 8, 8, 64)
    Q = rm.q(rm.nq(q)).view(2, 16, rm.h, rm.dk).transpose(1, 2)
    K = rm.k(rm.nk(tk)).view(2, -1, rm.h, rm.dk).transpose(1, 2)
    logits = (Q @ K.transpose(-2, -1)) / math.sqrt(rm.dk)
    lse = logits.reshape(2, rm.h, 16, 8, 8).logsumexp(-1)
    spread = {"over_rays": float(lse.std(2).max()), "over_views": float(lse.std(3).max()),
              "over_heads": float(lse.std(1).max()), "absmax": float(lse.abs().max())}
    note("l4b_lse_spread", {k: round(v, 4) for k, v in spread.items()})
    for k in ("over_rays", "over_views", "over_heads"):
        assert spread[k] > 1e-6, (k, spread[k])


# ==================================================== L5/L6 keys and parameters
def test_l5_exactly_one_new_state_dict_key_at_the_real_width():
    """At the width the cell actually trains (C=256), the switch must add exactly one tensor and
    nothing else -- no buffer, no bookkeeping parameter."""
    for nv in (2, 8):
        torch.manual_seed(0)
        film = OAAv2Depth(C=256, nviews=nv, geo_value_film=True)
        torch.manual_seed(0)
        both = OAAv2Depth(C=256, nviews=nv, geo_value_film=True, film_lse=True)
        added = sorted(set(both.state_dict()) - set(film.state_dict()))
        assert added == LSE_KEYS, (nv, added)
        assert sorted(set(film.state_dict()) - set(both.state_dict())) == []
        assert both.state_dict()["ray_mic.film_lse_w"].shape == (8,)
    note("l5_new_state_dict_keys", LSE_KEYS)


def test_l6_parameter_budget_is_plus_eight():
    """+8 parameters on top of GeoFiLM: one weight per head, no bias (value_film.bias already
    carries the per-head constant, so a bias here would be a duplicate direction). The ceiling
    still holds with EVERY switch on at the same time."""
    torch.manual_seed(0)
    nf = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, geo_value_film=True).parameters())
    torch.manual_seed(0)
    nl = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, geo_value_film=True,
                                           film_lse=True).parameters())
    assert nl - nf == 8, (nl - nf)
    torch.manual_seed(0)
    nb = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2).parameters())
    assert nl - nb == GEO_FEAT_DIM["raw"] * 8 + 8 + 8 == 56, (nl - nb)
    # the sh4 basis widens the geometry half only; the lse column stays 8 wide
    torch.manual_seed(0)
    nsh = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, geo_basis="sh4",
                                            geo_value_film=True).parameters())
    torch.manual_seed(0)
    nshl = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, geo_basis="sh4",
                                             geo_value_film=True, film_lse=True).parameters())
    assert nshl - nsh == 8, (nshl - nsh)
    torch.manual_seed(0)
    allon = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, fine_layerscale=1e-4, registers=4,
                                              scene_registers=True, geo_basis="sh4",
                                              geo_value_film=True, film_lse=True).parameters())
    note("l6_param_geofilm_M", f"{nf / 1e6:.6f}")
    note("l6_param_lse_added", nl - nf)
    note("l6_param_cell_M", f"{nl / 1e6:.6f}  (--stem-stride1 is free, it changes no parameter)")
    note("l6_param_every_switch_on_M", f"{allon / 1e6:.4f} (ceiling {PARAM_CEILING_M})")
    assert allon / 1e6 <= PARAM_CEILING_M, f"{allon / 1e6:.4f} M breaks the {PARAM_CEILING_M} M ceiling"


def test_l6b_film_lse_w_stays_out_of_the_param_lr_group():
    """Recorded as a deliberate choice, matching value_film: the new weight is zero-init but is
    NOT added to IDENTITY_INIT_PARAM_SUFFIXES, so --param-lr-mult does not see it. The queued cell
    leaves --param-lr-mult at its default 1.0, where the set is unused."""
    m = _small_model(2, geo_value_film=True, film_lse=True)
    hit = sorted(n for n, _ in m.named_parameters() if n.endswith(IDENTITY_INIT_PARAM_SUFFIXES))
    note("l6b_param_lr_group_with_lse", hit)
    assert not any("film_lse" in n for n in hit), hit


# ==================================================== L7 it is not inert when trained
def test_l7_a_nonzero_weight_moves_the_output_and_rezeroing_restores_it_exactly():
    """Otherwise L3 would be proving nothing: the column has to be capable of changing the
    prediction, and setting it back to its init has to restore the reference EXACTLY."""
    moved, restored = {}, 0.0
    for nv in (2, 4, 8):
        off = _small_model(nv, geo_value_film=True)
        on = _small_model(nv, geo_value_film=True, film_lse=True)
        assert _copy_shared(off, on) == LSE_KEYS
        x = _spec(2, nv)
        with torch.no_grad():
            torch.nn.init.normal_(on.ray_mic.film_lse_w, std=0.05)
        d = _maxdiff(off(x), on(x))
        moved[nv] = round(d, 6)
        assert d > 1e-6, (nv, "the lse column does nothing even with a real weight", d)
        with torch.no_grad():
            torch.nn.init.zeros_(on.ray_mic.film_lse_w)
        dz = _maxdiff(off(x), on(x))
        restored = max(restored, dz)
        assert dz == 0.0, (nv, dz)
    note("l7_output_shift_with_a_real_weight", moved)
    note("l7_restored_after_rezeroing", f"{restored:.3e} (must be exactly 0)")


def test_l7b_the_column_is_content_driven_which_is_the_whole_claim():
    """The distinguishing claim against --ray-gate: the multiplier must change when only the
    SPECTROGRAM changes, with geometry and poses held fixed. GeoFiLM alone cannot do this -- it is
    geometry-only -- so this test also separates the two cells."""
    nv = 4
    x1 = _spec(2, nv, seed=3)
    x2 = _spec(2, nv, seed=11)
    geo = _small_model(nv, geo_value_film=True)
    both = _small_model(nv, geo_value_film=True, film_lse=True)
    assert _copy_shared(geo, both) == LSE_KEYS
    with torch.no_grad():
        torch.nn.init.normal_(both.ray_mic.film_lse_w, std=0.05)

    def _film_stats(m, x):
        m.ray_mic.collect_stats = True
        m(x)
        m.ray_mic.collect_stats = False
        return m.ray_mic.stats["film_std"], m.ray_mic.stats["film_mean"]

    g1, gm1 = _film_stats(geo, x1)
    g2, gm2 = _film_stats(geo, x2)
    b1, bm1 = _film_stats(both, x1)
    b2, bm2 = _film_stats(both, x2)
    note("l7b_geofilm_film_std_two_inputs", (round(g1, 8), round(g2, 8)))
    note("l7b_with_lse_film_std_two_inputs", (round(b1, 6), round(b2, 6)))
    note("l7b_with_lse_film_mean_two_inputs", (round(bm1, 6), round(bm2, 6)))
    assert g1 == g2, "GeoFiLM alone must be input-independent; the test premise is broken"
    assert gm1 == gm2, "GeoFiLM alone must be input-independent; the test premise is broken"
    assert b1 != b2, "the lse column did not respond to the spectrogram"
    assert bm1 != bm2, "the lse column did not respond to the spectrogram"


def test_l7c_stats_report_the_multiplier_that_was_actually_applied():
    """--dump-attn-stats has to report the film INCLUDING the lse column, or M1/M3 would be read
    off the geometry half and the mechanism verdict would be about the wrong tensor. At init every
    reported number is still exactly 1.0 / 0.0, which is what makes L3 checkable from the stats."""
    rm = _ray_mod(geo_value_film=True, film_lse=True)
    rm.collect_stats = True
    q, tk, d3, poses = _ray_inputs(2, 16, 8, 8, 64)
    rm(q, tk, d3, poses, 8)
    assert rm.stats["film_mean"] == 1.0 and rm.stats["film_min"] == 1.0, rm.stats
    assert rm.stats["film_max"] == 1.0 and rm.stats["film_std"] == 0.0, rm.stats
    assert rm.stats["film_mean_per_view"].shape == (8,)
    note("l7c_stats_at_init", "mean/min/max = 1.0 exactly, std = 0.0 exactly, per_view shape (8,)")
    with torch.no_grad():
        torch.nn.init.normal_(rm.film_lse_w, std=0.05)
    rm(q, tk, d3, poses, 8)
    assert rm.stats["film_std"] > 0.0, rm.stats
    spread = float(rm.stats["film_mean_per_view"].std())
    note("l7c_stats_with_a_real_weight",
         {k: round(rm.stats[k], 4) for k in ("film_mean", "film_std", "film_min", "film_max")})
    note("l7c_film_mean_per_view_spread", round(spread, 6))
    assert spread > 1e-9, "film_mean_per_view is flat, so M1 could never be evaluated"


# ==================================================== L8 the one-layer rule
def test_l8_the_gate_is_one_affine_layer_not_two():
    """The literature review's general rule, encoded: a zero-init 2-layer head satisfies
    'multiplier exactly 1.0' but its INNER layer gets no gradient at step 0 (dL/dW_inner carries
    the zeroed outer weight as a factor). This gate must therefore be a single affine layer.
    Checked structurally -- one Parameter, 1-D, shape (heads,) -- and numerically by L4."""
    rm = _ray_mod(geo_value_film=True, film_lse=True)
    p = dict(rm.named_parameters())["film_lse_w"]
    assert p.dim() == 1 and p.shape == (8,), p.shape
    assert float(p.abs().max()) == 0.0, "film_lse_w must be zero-init"
    lse_mods = [n for n, mod in rm.named_modules()
                if "film_lse" in n and isinstance(mod, torch.nn.Module)]
    note("l8_film_lse_is_a_bare_parameter", f"dim={p.dim()} shape={tuple(p.shape)} "
                                            f"submodules={lse_mods}")
    assert lse_mods == [], lse_mods


# ==================================================== refusals
def test_film_lse_without_geo_value_film_is_refused():
    """The column widens GeoFiLM's film input. Without --geo-value-film there is no film, so the
    flag would silently do nothing -- the exact failure mode the project refuses by assertion."""
    try:
        _small_model(4, film_lse=True)
        raise RuntimeError("film_lse without geo_value_film should have been refused")
    except AssertionError as e:
        assert "geo-value-film" in str(e), str(e)
    note("film_lse_alone_refused", "--film-lse without --geo-value-film raises")


def test_incoherent_combinations_are_refused():
    """Everything GeoFiLM refuses, the pair must refuse too: --no-geo-bias removes the geometry
    features, hier / hier-geo already decompose the softmax per observation, --no-cross removes
    RayMicAttn outright."""
    cases = [("no_geo_bias", dict(no_geo_bias=True), "geo-bias"),
             ("hier", dict(view_fusion="hier"), "view_fusion"),
             ("hier-geo", dict(view_fusion="hier-geo"), "view_fusion"),
             ("no_cross", dict(no_cross=True), "no-cross")]
    for tag, kw, want in cases:
        try:
            _small_model(4, geo_value_film=True, film_lse=True, **kw)
            raise RuntimeError(f"film_lse + {tag} should have been refused")
        except AssertionError as e:
            assert want in str(e), (tag, str(e))
    note("lse_conflicts_refused", "no-geo-bias / hier / hier-geo / no-cross all raise")


def test_no_cross_refusal_names_film_lse():
    """--no-cross with ONLY --film-lse (no --geo-value-film) must still raise, and the message has
    to name the flag, or the next reader will think the combination is supported."""
    try:
        _small_model(4, no_cross=True, film_lse=True)
        raise RuntimeError("no_cross + film_lse should have been refused")
    except AssertionError as e:
        assert "film-lse" in str(e), str(e)
    note("no_cross_refusal_mentions_film_lse", True)


# ==================================================== run-level contracts
def test_forward_contract_for_the_queued_cell_config():
    """The exact configuration the cell queues (`--stem-stride1 --geo-value-film --film-lse`):
    (B,1,256,512) in [0,1], finite, and a live gradient on every new parameter."""
    for nv in (2, 4, 8):
        m = _small_model(nv, stem_stride1=True, geo_value_film=True, film_lse=True).train()
        D = m(_spec(2, nv))
        assert D.shape == (2, 1, 256, 512) and torch.isfinite(D).all(), nv
        assert float(D.min()) >= 0.0 and float(D.max()) <= 1.0, nv
        D.sum().backward()
        pm = dict(m.named_parameters())
        for q in BOTH_KEYS + ["ray_mic.q.weight", "ray_mic.bias_mlp.0.weight"]:
            g = pm[q].grad
            assert g is not None and torch.isfinite(g).all(), (nv, q)
            assert float(g.abs().sum()) > 0, (nv, q)
    note("lse_cell_forward_contract",
         "N=2/4/8 with --stem-stride1 --geo-value-film --film-lse: (2,1,256,512) in [0,1], "
         "live gradients")


def test_obs_mask_invariance_is_BROKEN_by_design_and_bounded():
    """Honest negative: GeoFiLM is geometry-only, so masking an observation leaves the prediction
    exactly unchanged. The lse column READS THE SPECTROGRAM, so a masked observation's content no
    longer influences the film only because its logits are forced to -inf -- which is exactly the
    behaviour we want and it IS preserved. What is NOT preserved is invariance to the CONTENT of a
    masked observation... except that masked logits are -inf before the logsumexp, so it is.
    This test pins that down, because if the lse were read before the mask the invariance that
    --subset-aug depends on would silently break."""
    nv = 8
    m = _small_model(nv, geo_value_film=True, film_lse=True)
    with torch.no_grad():
        torch.nn.init.normal_(m.ray_mic.film_lse_w, std=0.05)
        torch.nn.init.normal_(m.ray_mic.value_film.weight, std=0.05)
    ov = torch.ones(2, nv, dtype=torch.bool)
    ov[:, [1, 4, 6]] = False
    x = _spec(2, nv)
    a = m(x, obs_valid=ov)
    x2 = x.clone()
    x2[:, [1, 4, 6]] = torch.rand_like(x2[:, [1, 4, 6]]) * 7.0 + 3.0
    d = _maxdiff(a, m(x2, obs_valid=ov))
    note("lse_obs_mask_invariance", f"{d:.3e} (must be < 1e-12: the lse is read AFTER the mask)")
    assert d < 1e-12, d
    try:
        M._SHORTCIRCUIT_ALL_VALID = False
        dz = _maxdiff(m(x), m(x, obs_valid=torch.ones(2, nv, dtype=torch.bool)))
    finally:
        M._SHORTCIRCUIT_ALL_VALID = True
    note("lse_allvalid_identity", f"{dz:.3e}")
    assert dz < 1e-10, dz


def test_existing_hop16_checkpoint_strict_loads_into_default_off_model():
    """The default-off contract that matters operationally: a real trained checkpoint from before
    this change still loads with strict=True."""
    pth = os.path.join(CKPT_DIR, "best.pth")
    if not os.path.exists(pth):
        msg = f"checkpoint {pth} not found; set ARCHFIX_CKPT, or OAA_REF_OPTIONAL=1 to skip"
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
    note("hop16_ckpt_strict_load", f"{os.path.basename(CKPT_DIR)} loaded with strict=True "
                                   f"({len(sd)} tensors)")
    torch.manual_seed(0)
    mf = OAAv2Depth(C=256, nviews=2, stem_stride1=True, geo_value_film=True, film_lse=True)
    out = mf.load_state_dict(sd, strict=False)
    assert sorted(out.missing_keys) == BOTH_KEYS, out.missing_keys
    assert list(out.unexpected_keys) == [], out.unexpected_keys
    note("hop16_ckpt_warm_start_into_lse", "missing = film + lse tensors, unexpected = []")


def test_lse_config_round_trips_through_core_ckpt():
    """Save {state_dict, args} -> core.ckpt.build(args, DM) -> strict load -> identical prediction.
    Without this the cell trains fine and is then UNSCOREABLE: eval.py catches the strict-load
    failure per run and prints only `[skip <run>]`, so the OAA row vanishes from the table."""
    from core.ckpt import build
    DM = types.SimpleNamespace(IN_CH={"r2": 2, "cB": 4, "fb": 4, "r6": 6, "r8": 8}, POSES={},
                               WINDOW=0, HOP=160, WIN=400, N_FFT=512)
    DM.__name__ = "data_0422"
    cfgs = [{}, {"geo_value_film": True},
            {"geo_value_film": True, "film_lse": True},
            {"geo_value_film": True, "film_lse": True, "geo_basis": "sh4"},
            {"geo_value_film": True, "film_lse": True, "qk_norm": True},
            {"geo_value_film": True, "film_lse": True, "registers": 4, "scene_registers": True,
             "fine_layerscale": 1e-4, "geo_basis": "sh4"}]
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
                         geo_basis=extra.get("geo_basis", "raw"),
                         geo_value_film=extra.get("geo_value_film", False),
                         film_lse=extra.get("film_lse", False)).eval()
        rebuilt, dmode, nch, kind, poses = build(args, DM)
        assert set(rebuilt.state_dict()) == set(src.state_dict()), extra
        rebuilt.load_state_dict(src.state_dict(), strict=True)
        rebuilt.eval()
        x = torch.rand(1, 2, 24, 40)
        with torch.no_grad():
            d = _maxdiff(src(x), rebuilt(x))
        assert d == 0.0, (extra, d)
    note("lse_ckpt_round_trip", f"{len(cfgs)} configs strict-loaded and prediction-compared: "
                                f"max|diff| = 0.0")


def test_train_oaa_declares_the_flag_and_wires_it_to_the_model():
    """The flag has to reach the model, and `train_oaa.py` builds its parser inside `main()`, so it
    cannot be imported and introspected. The next best honest check is the source contract: the
    argument is declared as a store_true (hence default False) and it is passed to OAAv2Depth. A
    flag that parses but is never forwarded is the silent-no-op failure this repo refuses."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(repo, "train_oaa.py")).read()
    decl = '    p.add_argument("--film-lse", action="store_true")'
    assert src.count(decl) == 1, "the --film-lse declaration is missing or duplicated"
    assert src.count("film_lse=a.film_lse") == 1, "--film-lse is parsed but never reaches the model"
    assert "default=" not in decl, "store_true already defaults to False; an explicit default " \
                                   "would be a second source of truth"
    note("train_oaa_flag_source_contract",
         "--film-lse declared store_true once and forwarded as film_lse=a.film_lse once")


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
