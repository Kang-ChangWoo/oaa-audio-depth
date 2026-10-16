"""Equivalence gate for GeoFiLM (`--geo-value-film`).

Run either way:
    python tests/test_oaa_geofilm.py          # prints every measured number
    pytest -q tests/test_oaa_geofilm.py

What is under test, in one sentence: geometry features multiply the POST-softmax attention
probabilities per (ray, observation, head), so the model can decide how *hard* to read an
observation and not only which token wins. `nn.Linear(gf, heads)` with zero weight AND zero bias,
read as `1 + Linear(feats)`, so the multiplier is exactly 1.0 at init.

Same discipline as `tests/test_oaa_arch_fixes.py`: every *equivalence* claim runs on CPU in
float64, because the CUDA TF32/flash kernels are not bit-reproducible and a GPU run could only
establish a loose bound. Gradient differences are divided by the LARGEST gradient in the
configuration, never by each tensor's own max.

`OAA_REF_MODULE_435FC60` points at a pristine `435fc60` copy of model/oaa.py (default
`/root/local1/changwoo/archfix_20261016/ref/oaa_ref_435fc60.py`, produced with
`git show 435fc60:model/oaa.py`). `435fc60` is the archfix tip, i.e. the parent of this change --
so "the flag off reproduces 435fc60 bit for bit" is the claim, not "it reproduces eda3871".
Missing -> the parity tests FAIL rather than silently pass, unless OAA_REF_OPTIONAL=1 records an
explicit, counted skip.

`ARCHFIX_CKPT` points at a run directory whose best.pth must strict-load into the default-off
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
from model.oaa import (OAAv2Depth, RayMicAttn, GEO_FEAT_DIM,            # noqa: E402
                       IDENTITY_INIT_PARAM_SUFFIXES)

REF_PATH = os.environ.get("OAA_REF_MODULE_435FC60",
                          "/root/local1/changwoo/archfix_20261016/ref/oaa_ref_435fc60.py")
CKPT_DIR = os.environ.get(
    "ARCHFIX_CKPT",
    "/root/local1/changwoo/relevel_hop16_obs48_20261015/runs/rep/relevel_h16_stem1_r2_rep_s0")
SMALL = dict(C=64, lh=4, lw=8, enc_res=(16, 32))     # 256x512 output, tiny attention
PARAM_CEILING_M = 15.45                              # PREREG hard ceiling; the paper's #P prints 15.4
FILM_KEYS = ["ray_mic.value_film.bias", "ray_mic.value_film.weight"]
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
        msg = (f"pristine 435fc60 module not found at {REF_PATH}.\n"
               f"    git show 435fc60:model/oaa.py > {REF_PATH}\n"
               f"    or set OAA_REF_OPTIONAL=1 for an explicit, counted skip.")
        if os.environ.get("OAA_REF_OPTIONAL") == "1":
            raise _Skipped(msg)
        raise FileNotFoundError(msg)
    spec = importlib.util.spec_from_file_location("oaa_ref_435fc60", REF_PATH)
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


# ============================================================ G1/G2 the flag off == 435fc60
def test_g1_flag_off_is_bit_identical_to_435fc60():
    """The patch must be completely inert when `--geo-value-film` is not passed: same forward, same
    gradients, same state_dict keys as the pristine archfix tip. Exactly 0.0, float64, N = 2/4/8."""
    ref = _ref_module()
    worst_f = worst_g = 0.0
    for nv in (2, 4, 8):
        r = _small_model(nv, ref=ref).train()
        o = _small_model(nv).train()
        added = _copy_shared(r, o)
        assert added == [], f"flag OFF must add no keys, got {added}"
        assert sorted(r.state_dict()) == sorted(o.state_dict())
        x = _spec(2, nv)
        Dr, gr = _grads_of(r, x)
        Do, go = _grads_of(o, x)
        df = _maxdiff(Dr, Do)
        dg, where, _ = _cmp_grads(gr, go, f"N={nv}")
        worst_f, worst_g = max(worst_f, df), max(worst_g, dg)
        assert df == 0.0, (nv, df)
        assert dg == 0.0, (nv, dg, where)
    note("g1_off_vs_435fc60_forward_maxdiff", f"{worst_f:.3e} (N=2/4/8, must be exactly 0)")
    note("g2_off_vs_435fc60_grad_maxrel", f"{worst_g:.3e} (must be exactly 0)")


def test_g1b_flag_off_masked_and_register_branches_match_435fc60():
    """The patch splits all THREE global-softmax sub-branches, not just the unmasked one. Each one
    has to stay bit-identical with the flag off, or the inertness claim covers only one code path."""
    ref = _ref_module()
    worst = 0.0
    for nv in (2, 4, 8):
        for tag, kw, ov in (("masked", {}, True), ("registers", dict(registers=4), False),
                            ("masked+registers", dict(registers=4), True)):
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
    note("g1b_off_all_three_branches_maxdiff",
         f"{worst:.3e} (9 configurations x 3 sub-branches, must be exactly 0)")


# ============================================================ G3 the flag ON, at init
def test_g3_film_on_at_init_is_bit_identical():
    """film = 1 + Linear(feats) with zero weight AND zero bias is exactly 1.0, and p * 1.0 is exact
    in IEEE-754, so turning the switch ON must not move a single bit at init. This is the whole
    reason the scale is folded into the probabilities instead of into V."""
    worst_f = worst_g = 0.0
    for nv in (2, 4, 8):
        off = _small_model(nv).train()
        on = _small_model(nv, geo_value_film=True).train()
        added = _copy_shared(off, on)
        assert added == FILM_KEYS, added
        x = _spec(2, nv)
        Do, go = _grads_of(off, x)
        Dn, gn = _grads_of(on, x)
        df = _maxdiff(Do, Dn)
        dg, where, _ = _cmp_grads(go, {k: v for k, v in gn.items() if k in go}, f"N={nv}")
        worst_f, worst_g = max(worst_f, df), max(worst_g, dg)
        assert df == 0.0, (nv, df)
        assert dg == 0.0, (nv, dg, where)
    note("g3_film_on_at_init_forward_maxdiff", f"{worst_f:.3e} (N=2/4/8, must be exactly 0)")
    note("g3_film_on_at_init_shared_grad_maxrel", f"{worst_g:.3e} (must be exactly 0)")


def test_g3b_film_on_at_init_in_the_masked_and_register_branches():
    """Closes the gap the out-of-repo copy recorded as [미확인]: no test had driven the masked
    branch with a genuinely dropped observation AND the film on. Branch 3 (registers) is covered
    too -- registers belong to no observation and are deliberately NOT filmed."""
    worst = 0.0
    for nv in (2, 4, 8):
        for tag, kw in (("masked", {}), ("masked+registers", dict(registers=4))):
            off = _small_model(nv, **kw)
            on = _small_model(nv, geo_value_film=True, **kw)
            assert _copy_shared(off, on) == FILM_KEYS
            x = _spec(2, nv)
            obs = torch.ones(2, nv, dtype=torch.bool)
            obs[:, 1] = False
            if nv >= 4:
                obs[0, 2] = False
            d = _maxdiff(off(x, obs_valid=obs), on(x, obs_valid=obs))
            worst = max(worst, d)
            assert d == 0.0, (nv, tag, d)
        # and the pure register branch, with the all-valid short circuit disabled
        off = _small_model(nv, registers=4)
        on = _small_model(nv, registers=4, geo_value_film=True)
        assert _copy_shared(off, on) == FILM_KEYS
        x = _spec(2, nv)
        d = _maxdiff(off(x), on(x))
        worst = max(worst, d)
        assert d == 0.0, (nv, "registers", d)
    note("g3b_film_on_masked_register_branches_maxdiff",
         f"{worst:.3e} (N=2/4/8 x masked / masked+registers / registers, must be exactly 0)")


def test_g3c_reference_state_dict_loads_with_only_the_two_new_keys_missing():
    """An existing checkpoint has to load into the film-on model: missing exactly the two new
    tensors, nothing unexpected. That is what makes the switch deployable on a warm start."""
    for nv in (2, 8):
        off = _small_model(nv)
        on = _small_model(nv, geo_value_film=True)
        out = on.load_state_dict(off.state_dict(), strict=False)
        assert sorted(out.missing_keys) == FILM_KEYS, out.missing_keys
        assert list(out.unexpected_keys) == [], out.unexpected_keys
    note("g3c_loadstate_missing", FILM_KEYS)
    note("g3c_loadstate_unexpected", [])


# ============================================================ G4 the gradient is alive
def test_g4_value_film_gets_a_gradient_at_step_0():
    """archfix B1's lesson, applied as a gate: a zero-init weight does NOT kill its own gradient
    (dL/dW = g (x) x), but a zero-init multiplicative GAMMA does. GeoFiLM is the first kind, so the
    module must be learnable from step 0. If this is 0 the switch is cosmetic."""
    rows = {}
    for nv in (2, 4, 8):
        m = _small_model(nv, geo_value_film=True).train()
        _, g = _grads_of(m, _spec(2, nv))
        gw = float(g["ray_mic.value_film.weight"].abs().max())
        gb = float(g["ray_mic.value_film.bias"].abs().max())
        rows[nv] = (round(gw, 4), round(gb, 4))
        assert gw > 0.0, (nv, "value_film.weight gradient is dead")
        assert gb > 0.0, (nv, "value_film.bias gradient is dead")
    note("g4_value_film_grad_absmax_weight_bias", rows)


# ============================================================ G5/G6 keys and parameters
def test_g5_only_two_new_state_dict_keys_at_the_real_width():
    """At the width the cells actually train (C=256), the switch must add exactly two tensors and
    nothing else -- no buffer, no bookkeeping parameter."""
    for nv in (2, 8):
        torch.manual_seed(0)
        base = OAAv2Depth(C=256, nviews=nv)
        torch.manual_seed(0)
        film = OAAv2Depth(C=256, nviews=nv, geo_value_film=True)
        added = sorted(set(film.state_dict()) - set(base.state_dict()))
        assert added == FILM_KEYS, (nv, added)
        assert sorted(set(base.state_dict()) - set(film.state_dict())) == []
    note("g5_new_state_dict_keys", FILM_KEYS)


def test_g6_parameter_budget_with_the_film_on():
    """+48 parameters at the `raw` basis (Linear(5, 8) = 40 + 8), and the ceiling still holds with
    EVERY archfix switch on at the same time. Over 15.45 M means the implementation is wrong."""
    torch.manual_seed(0)
    nb = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2).parameters())
    torch.manual_seed(0)
    nf = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, geo_value_film=True).parameters())
    gf = GEO_FEAT_DIM["raw"]
    assert nf - nb == gf * 8 + 8 == 48, (nf - nb, gf)
    # the sh4 basis widens the film input the same way it widens the bias MLP input
    torch.manual_seed(0)
    nsh = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, geo_basis="sh4",
                                            geo_value_film=True).parameters())
    torch.manual_seed(0)
    nshonly = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, geo_basis="sh4").parameters())
    assert nsh - nshonly == GEO_FEAT_DIM["sh4"] * 8 + 8 == 224, (nsh - nshonly)
    torch.manual_seed(0)
    allon = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, fine_layerscale=1e-4, registers=4,
                                              scene_registers=True, geo_basis="sh4",
                                              geo_value_film=True).parameters())
    note("g6_param_base_M", f"{nb / 1e6:.4f}")
    note("g6_param_film_only_added", nf - nb)
    note("g6_param_film_sh4_added", nsh - nshonly)
    note("g6_param_every_switch_on_M", f"{allon / 1e6:.4f} (ceiling {PARAM_CEILING_M})")
    assert allon / 1e6 <= PARAM_CEILING_M, f"{allon / 1e6:.4f} M breaks the {PARAM_CEILING_M} M ceiling"


def test_g6b_value_film_stays_out_of_the_param_lr_group():
    """Recorded as a deliberate choice, not an oversight: `value_film` is zero-init like
    `scene_mlp`, but it is NOT added to IDENTITY_INIT_PARAM_SUFFIXES, so --param-lr-mult does not
    see it. The queued cells leave --param-lr-mult at its default 1.0, where the set is unused."""
    m = _small_model(2, geo_value_film=True)
    hit = sorted(n for n, _ in m.named_parameters() if n.endswith(IDENTITY_INIT_PARAM_SUFFIXES))
    note("g6b_param_lr_group_with_film", hit)
    assert not any("value_film" in n for n in hit), hit


# ============================================================ G7 it is not inert when trained
def test_g7_a_nonzero_film_moves_the_output_and_zeroing_restores_it_exactly():
    """Otherwise G3 would be proving nothing: the switch has to be capable of changing the
    prediction, and setting the film back to its init has to restore the reference EXACTLY."""
    moved, restored = {}, 0.0
    for nv in (2, 4, 8):
        off = _small_model(nv)
        on = _small_model(nv, geo_value_film=True)
        assert _copy_shared(off, on) == FILM_KEYS
        x = _spec(2, nv)
        with torch.no_grad():
            torch.nn.init.normal_(on.ray_mic.value_film.weight, std=0.05)
            torch.nn.init.normal_(on.ray_mic.value_film.bias, std=0.05)
        d = _maxdiff(off(x), on(x))
        moved[nv] = round(d, 6)
        assert d > 1e-6, (nv, "the film does nothing even with real weights", d)
        with torch.no_grad():
            torch.nn.init.zeros_(on.ray_mic.value_film.weight)
            torch.nn.init.zeros_(on.ray_mic.value_film.bias)
        dz = _maxdiff(off(x), on(x))
        restored = max(restored, dz)
        assert dz == 0.0, (nv, dz)
    note("g7_film_output_shift_with_real_weights", moved)
    note("g7_restored_after_rezeroing", f"{restored:.3e} (must be exactly 0)")


def test_g7b_film_is_per_ray_per_observation_per_head_and_reported():
    """The mechanism claim, checked on the module rather than argued: the multiplier has to vary
    along rays, along observations and along heads, and --dump-attn-stats has to report it so the
    cell can be read mechanistically and not only by its RMSE."""
    rm = _ray_mod(geo_value_film=True)
    with torch.no_grad():
        torch.nn.init.normal_(rm.value_film.weight, std=0.3)
        torch.nn.init.normal_(rm.value_film.bias, std=0.1)
    rm.collect_stats = True
    q, tk, d3, poses = _ray_inputs(2, 16, 8, 8, 64)
    rm(q, tk, d3, poses, 8)
    feats = rm._geo_basis_feats(d3, poses)                                 # (R,N,gf)
    fv = 1.0 + rm.value_film(feats)                                        # (R,N,h)
    assert fv.shape == (16, 8, 8), fv.shape
    spread = {"over_rays": float(fv.std(0).max()), "over_views": float(fv.std(1).max()),
              "over_heads": float(fv.std(2).max())}
    note("g7b_film_spread", {k: round(v, 4) for k, v in spread.items()})
    for k, v in spread.items():
        assert v > 1e-6, (k, v)
    for k in ("film_mean", "film_std", "film_min", "film_max"):
        assert k in rm.stats, k
    assert rm.stats["film_mean_per_view"].shape == (8,)
    note("g7b_film_stats_at_a_trained_film",
         {k: round(rm.stats[k], 4) for k in ("film_mean", "film_std", "film_min", "film_max")})
    # and at init every one of those is exactly 1.0 / 0.0
    rm0 = _ray_mod(geo_value_film=True)
    rm0.collect_stats = True
    rm0(q, tk, d3, poses, 8)
    assert rm0.stats["film_mean"] == 1.0 and rm0.stats["film_min"] == 1.0, rm0.stats
    assert rm0.stats["film_max"] == 1.0 and rm0.stats["film_std"] == 0.0, rm0.stats
    note("g7b_film_stats_at_init", "mean/min/max = 1.0 exactly, std = 0.0 exactly")


# ============================================================ refusals
def test_incoherent_combinations_are_refused():
    """The film reads the geometry features and reweights the single global softmax. With
    --no-geo-bias those features are never computed; with hier / hier-geo the softmax is already
    decomposed per observation; with --no-cross RayMicAttn does not exist. All three must raise
    instead of silently doing nothing or guessing a semantics."""
    cases = [("no_geo_bias", dict(no_geo_bias=True), "geo-bias"),
             ("hier", dict(view_fusion="hier"), "view_fusion"),
             ("hier-geo", dict(view_fusion="hier-geo"), "view_fusion"),
             ("no_cross", dict(no_cross=True), "no-cross")]
    for tag, kw, want in cases:
        try:
            _small_model(4, geo_value_film=True, **kw)
            raise RuntimeError(f"geo_value_film + {tag} should have been refused")
        except AssertionError as e:
            assert want in str(e), (tag, str(e))
    note("film_conflicts_refused", "no-geo-bias / hier / hier-geo / no-cross all raise")


# ============================================================ run-level contracts
def test_forward_contract_for_the_queued_cell_config():
    """The exact configuration the 3 seeds queue (`--stem-stride1 --geo-value-film`): (B,1,256,512)
    in [0,1], finite, and a live gradient on every new parameter."""
    for nv in (2, 4, 8):
        m = _small_model(nv, stem_stride1=True, geo_value_film=True).train()
        D = m(_spec(2, nv))
        assert D.shape == (2, 1, 256, 512) and torch.isfinite(D).all(), nv
        assert float(D.min()) >= 0.0 and float(D.max()) <= 1.0, nv
        D.sum().backward()
        pm = dict(m.named_parameters())
        for q in FILM_KEYS + ["ray_mic.q.weight", "ray_mic.bias_mlp.0.weight"]:
            g = pm[q].grad
            assert g is not None and torch.isfinite(g).all(), (nv, q)
            assert float(g.abs().sum()) > 0, (nv, q)
    note("film_cell_forward_contract",
         "N=2/4/8 with --stem-stride1 --geo-value-film: (2,1,256,512) in [0,1], live gradients")


def test_obs_mask_invariance_survives_a_trained_film():
    """With a real (not zero) film, an invalid observation still must not change the prediction.
    The film is driven by geometry only -- it never reads the spectrogram -- so masking stays
    exact. This is the invariance `--subset-aug` depends on."""
    nv = 8
    m = _small_model(nv, geo_value_film=True)
    with torch.no_grad():
        torch.nn.init.normal_(m.ray_mic.value_film.weight, std=0.05)
        torch.nn.init.normal_(m.ray_mic.value_film.bias, std=0.05)
    ov = torch.ones(2, nv, dtype=torch.bool); ov[:, [1, 4, 6]] = False
    x = _spec(2, nv)
    a = m(x, obs_valid=ov)
    x2 = x.clone()
    x2[:, [1, 4, 6]] = torch.rand_like(x2[:, [1, 4, 6]]) * 7.0 + 3.0
    d = _maxdiff(a, m(x2, obs_valid=ov))
    note("film_obs_mask_invariance", f"{d:.3e}")
    assert d < 1e-12, d
    try:
        M._SHORTCIRCUIT_ALL_VALID = False
        dz = _maxdiff(m(x), m(x, obs_valid=torch.ones(2, nv, dtype=torch.bool)))
    finally:
        M._SHORTCIRCUIT_ALL_VALID = True
    note("film_allvalid_identity", f"{dz:.3e}")
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
    # and the same checkpoint into the film-on model leaves exactly the two new keys missing
    torch.manual_seed(0)
    mf = OAAv2Depth(C=256, nviews=2, stem_stride1=True, geo_value_film=True)
    out = mf.load_state_dict(sd, strict=False)
    assert sorted(out.missing_keys) == FILM_KEYS, out.missing_keys
    assert list(out.unexpected_keys) == [], out.unexpected_keys
    note("hop16_ckpt_warm_start_into_film", "missing = the two film tensors, unexpected = []")


def test_film_config_round_trips_through_core_ckpt():
    """Save {state_dict, args} -> core.ckpt.build(args, DM) -> strict load -> identical prediction.
    Without this the cell trains fine and is then UNSCOREABLE: eval.py catches the strict-load
    failure per run and prints only `[skip <run>]`, so the OAA row vanishes and the baselines look
    like the whole table."""
    from core.ckpt import build
    DM = types.SimpleNamespace(IN_CH={"r2": 2, "cB": 4, "fb": 4, "r6": 6, "r8": 8}, POSES={},
                               WINDOW=0, HOP=160, WIN=400, N_FFT=512)
    DM.__name__ = "data_0422"
    cfgs = [{}, {"geo_value_film": True}, {"geo_value_film": True, "geo_basis": "sh4"},
            {"geo_value_film": True, "qk_norm": True},
            {"geo_value_film": True, "registers": 4, "scene_registers": True,
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
                         geo_value_film=extra.get("geo_value_film", False)).eval()
        rebuilt, dmode, nch, kind, poses = build(args, DM)
        assert set(rebuilt.state_dict()) == set(src.state_dict()), extra
        rebuilt.load_state_dict(src.state_dict(), strict=True)
        rebuilt.eval()
        x = torch.rand(1, 2, 24, 40)
        with torch.no_grad():
            d = _maxdiff(src(x), rebuilt(x))
        assert d == 0.0, (extra, d)
    note("film_ckpt_round_trip", f"{len(cfgs)} configs strict-loaded and prediction-compared: "
                                 f"max|diff| = 0.0")


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
