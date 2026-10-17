"""Equivalence gate for the two mid-band structure variants, 2026-10-17.

Run either way:
    python tests/test_oaa_midband.py          # prints every measured number
    pytest -q tests/test_oaa_midband.py

What is under test, in two sentences:

  `--pose-reinject`  re-adds the per-observation pose code before each `InterMicAttn` block -- the
      one per-observation site the reference never gives it -- through one zero-initialised
      per-channel gamma per fusion round, so `x + gamma * pose_code` adds exactly 0.0 at init.
      A third site, the fine ray-lift query, was dropped after this gate measured its gradient at
      exactly 0.0 at step 0: it sits behind the zero-init `fine_to_dec` conv. See
      `test_g3b_the_dropped_fine_site_is_why_it_was_dropped`.
  `--geo-ray-gain`   multiplies the cross-attention OUTPUT by a per-(ray, head) scalar predicted
      OUTSIDE the softmax from [observation-mean geometry features, ray direction], via one
      zero-initialised `nn.Linear` read as `1 + Linear(x)`, so the gain is exactly 1.0 at init.
      It is applied after the sum over observations and tokens, so unlike GeoFiLM it can change
      only the TOTAL magnitude of a ray and cannot redistribute mass between observations.

Same discipline as `tests/test_oaa_geofilm.py`: every *equivalence* claim runs on CPU in float64,
because the CUDA TF32/flash kernels are not bit-reproducible and a GPU run could only establish a
loose bound. Gradient differences are divided by the LARGEST gradient in the configuration, never
by each tensor's own max.

`OAA_REF_MODULE_D2A95A6` points at a pristine `d2a95a6` copy of model/oaa.py (default
`/root/local1/changwoo/archfix_20261016/ref/oaa_ref_d2a95a6.py`, produced with
`git show d2a95a6:model/oaa.py`). `d2a95a6` is the parent of this change and the commit every
control seed was trained at, so "the flags off reproduce d2a95a6 bit for bit" is the claim.
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

REF_PATH = os.environ.get("OAA_REF_MODULE_D2A95A6",
                          "/root/local1/changwoo/archfix_20261016/ref/oaa_ref_d2a95a6.py")
CKPT_DIR = os.environ.get(
    "ARCHFIX_CKPT",
    "/root/local1/changwoo/relevel_hop16_obs48_20261015/runs/rep/relevel_h16_stem1_r2_rep_s0")
SMALL = dict(C=64, lh=4, lw=8, enc_res=(16, 32))     # 256x512 output, tiny attention
PARAM_CEILING_M = 15.45                              # PREREG hard ceiling; the paper's #P prints 15.4
POSE_KEYS = ["pose_rein.0", "pose_rein.1"]                        # at the default rounds=2
GAIN_KEYS = ["ray_mic.ray_gain.bias", "ray_mic.ray_gain.weight"]
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
        msg = (f"pristine d2a95a6 module not found at {REF_PATH}.\n"
               f"    git show d2a95a6:model/oaa.py > {REF_PATH}\n"
               f"    or set OAA_REF_OPTIONAL=1 for an explicit, counted skip.")
        if os.environ.get("OAA_REF_OPTIONAL") == "1":
            raise _Skipped(msg)
        raise FileNotFoundError(msg)
    spec = importlib.util.spec_from_file_location("oaa_ref_d2a95a6", REF_PATH)
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


ON = [("pose", dict(pose_reinject=True), POSE_KEYS),
      ("gain", dict(geo_ray_gain=True), GAIN_KEYS),
      ("both", dict(pose_reinject=True, geo_ray_gain=True), sorted(POSE_KEYS + GAIN_KEYS))]


def _set_nonzero(m, std=0.05, seed=11):
    """Give every newly added parameter a real (non-identity) value."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n in POSE_KEYS or n in GAIN_KEYS:
                torch.nn.init.normal_(p, std=std)


def _rezero(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n in POSE_KEYS or n in GAIN_KEYS:
                torch.nn.init.zeros_(p)


# ============================================================ G1 both flags off == d2a95a6
def test_g1_flags_off_are_bit_identical_to_d2a95a6():
    """The patch must be completely inert when neither switch is passed: same forward, same
    gradients, same state_dict keys as the pristine parent commit. Exactly 0.0, float64, N=2/4/8."""
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
    note("g1_off_vs_d2a95a6_forward_maxdiff", f"{worst_f:.3e} (N=2/4/8, must be exactly 0)")
    note("g1_off_vs_d2a95a6_grad_maxrel", f"{worst_g:.3e} (must be exactly 0)")


def test_g1b_flags_off_in_every_sub_branch():
    """`--geo-ray-gain` touches the one line both fusion modes funnel through, and `--pose-reinject`
    touches the masked fine-skip mean. With the flags off all of masked / registers / hier /
    hier-geo have to stay bit-identical, or the inertness claim covers one code path only."""
    ref = _ref_module()
    worst = 0.0
    for nv in (2, 4, 8):
        for tag, kw, ov in (("masked", {}, True),
                            ("registers", dict(registers=4), False),
                            ("masked+registers", dict(registers=4), True),
                            ("hier", dict(view_fusion="hier"), False),
                            ("hier-geo", dict(view_fusion="hier-geo"), False),
                            ("hier+masked", dict(view_fusion="hier"), True)):
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


# ============================================================ G2 the flags ON, at init
def test_g2_each_flag_on_at_init_is_bit_identical():
    """gamma = 0 makes the pose term exactly 0.0 and `x + 0.0 == x`; zero weight AND bias make the
    gain exactly 1.0 and `x * 1.0 == x` exactly in IEEE-754. So turning either switch on -- or both
    -- must not move a single bit at init, forward or in any shared gradient."""
    rows = {}
    for tag, kw, keys in ON:
        worst_f = worst_g = 0.0
        for nv in (2, 4, 8):
            off = _small_model(nv).train()
            on = _small_model(nv, **kw).train()
            added = _copy_shared(off, on)
            assert added == keys, (tag, nv, added)
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
    """Same claim in the masked / register / hier branches, with an observation genuinely dropped.
    `--pose-reinject` writes into the fine-skip query, which is exactly where the masked mean runs,
    so the masked branch is the one that could have been missed."""
    worst = 0.0
    for tag, kw, keys in ON:
        for nv in (2, 4, 8):
            for btag, bkw, ov in (("masked", {}, True),
                                  ("masked+registers", dict(registers=4), True),
                                  ("registers", dict(registers=4), False),
                                  ("hier", dict(view_fusion="hier"), False),
                                  ("hier+masked", dict(view_fusion="hier"), True),
                                  ("hier-geo", dict(view_fusion="hier-geo"), False)):
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
                assert d == 0.0, (tag, btag, nv, d)
    note("g2b_on_at_init_all_sub_branches_maxdiff",
         f"{worst:.3e} (3 switch settings x N=2/4/8 x 6 sub-branches, must be exactly 0)")


def test_g2c_rounds_other_than_two():
    """`pose_rein` is one gamma per fusion round, indexed by the loop counter. rounds=1 and rounds=3
    must build the right number of tensors and stay bit-identical at init -- an off-by-one here
    would silently re-inject the wrong round's gamma."""
    for rounds in (1, 3):
        keys = [f"pose_rein.{i}" for i in range(rounds)]
        for nv in (2, 8):
            off = _small_model(nv, rounds=rounds)
            on = _small_model(nv, rounds=rounds, pose_reinject=True)
            assert _copy_shared(off, on) == sorted(keys), (rounds, nv)
            d = _maxdiff(off(_spec(2, nv)), on(_spec(2, nv)))
            assert d == 0.0, (rounds, nv, d)
    note("g2c_rounds_1_and_3_keys_and_maxdiff", "one gamma per round, max|diff| = 0.0")


# ============================================================ G3 the gradients are alive
def test_g3_every_new_parameter_has_a_live_gradient_at_step_0():
    """A zero-init affine keeps its own gradient (dL/dW = g (x) x) but a zero-init multiplicative
    gamma on a cold branch does not. Both variants are the first kind by construction; this is the
    gate that proves it rather than arguing it. A zero here means the switch is cosmetic."""
    rows = {}
    for tag, kw, keys in ON:
        for nv in (2, 4, 8):
            m = _small_model(nv, **kw).train()
            _, g = _grads_of(m, _spec(2, nv))
            for k in keys:
                a = float(g[k].abs().max())
                rows[f"{tag}/N{nv}/{k}"] = round(a, 4)
                assert a > 0.0, (tag, nv, k, "gradient is dead")
    note("g3_new_param_grad_absmax", rows)


def test_g3b_the_dropped_fine_site_is_why_it_was_dropped():
    """The adverse measurement that shaped the design, kept as a live test so the reason cannot be
    lost. A gamma on the fine ray-lift query would be bit-identical at init just like the inter
    gammas, but `fine_to_dec` is zero-initialised, so the gradient reaching anything inside the fine
    branch is exactly W^T g = 0 at step 0. Stand-in for that gamma: `fine_q` itself, which is the
    tensor the pose term would have been added to. If this ever becomes non-zero, the fine site has
    become testable and the variant should be revisited together with --fine-layerscale."""
    rows = {}
    for nv in (2, 4, 8):
        m = _small_model(nv, pose_reinject=True).train()
        _, g = _grads_of(m, _spec(2, nv))
        dead = float(g["fine_q"].abs().max())
        live = float(g["pose_rein.0"].abs().max())
        rows[nv] = {"fine_branch_grad_absmax": dead, "inter_site_grad_absmax": round(live, 4)}
        assert dead == 0.0, (nv, "the fine branch is no longer cold; revisit the dropped site", dead)
        assert live > 0.0, (nv, live)
        # and with the archfix repair in place the same branch is alive -- that is the combination
        # under which the dropped site would become gateable
        m2 = _small_model(nv, pose_reinject=True, fine_layerscale=0.1).train()
        _, g2 = _grads_of(m2, _spec(2, nv))
        warm = float(g2["fine_q"].abs().max())
        rows[nv]["fine_branch_grad_with_layerscale"] = round(warm, 4)
        assert warm > 0.0, (nv, warm)
    note("g3b_fine_site_cold_start", rows)


# ============================================================ G4 parameter counts
def test_g4_exact_parameter_counts_at_the_real_width():
    """Exact integers, not "about". pose_reinject = rounds * dim; geo_ray_gain =
    (gf + 3) * heads + heads. The ceiling must still hold with every switch on at once."""
    torch.manual_seed(0)
    nb = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2).parameters())
    counts = {}
    for tag, kw, _ in ON:
        torch.manual_seed(0)
        n = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, **kw).parameters())
        counts[tag] = n - nb
    assert counts["pose"] == 2 * 256 == 512, counts
    assert counts["gain"] == (GEO_FEAT_DIM["raw"] + 3) * 8 + 8 == 72, counts
    assert counts["both"] == 512 + 72 == 584, counts
    # the sh4 basis widens the gain input exactly as it widens the bias MLP input
    torch.manual_seed(0)
    nsh = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, geo_basis="sh4").parameters())
    torch.manual_seed(0)
    nshg = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, geo_basis="sh4",
                                             geo_ray_gain=True).parameters())
    assert nshg - nsh == (GEO_FEAT_DIM["sh4"] + 3) * 8 + 8 == 248, (nshg - nsh)
    # rounds scales the pose side and only the pose side
    torch.manual_seed(0)
    n3 = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, rounds=3).parameters())
    torch.manual_seed(0)
    n3p = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, rounds=3,
                                            pose_reinject=True).parameters())
    assert n3p - n3 == 3 * 256 == 768, (n3p - n3)
    torch.manual_seed(0)
    allon = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, fine_layerscale=1e-4, registers=4,
                                              scene_registers=True, geo_basis="sh4",
                                              geo_value_film=True, film_lse=True,
                                              pose_reinject=True,
                                              geo_ray_gain=True).parameters())
    note("g4_param_base_M", f"{nb / 1e6:.6f}")
    note("g4_param_added", counts)
    note("g4_param_added_gain_sh4", nshg - nsh)
    note("g4_param_added_pose_rounds3", n3p - n3)
    note("g4_param_every_switch_on_M", f"{allon / 1e6:.4f} (ceiling {PARAM_CEILING_M})")
    assert allon / 1e6 <= PARAM_CEILING_M, f"{allon / 1e6:.4f} M breaks the {PARAM_CEILING_M} M ceiling"


def test_g4b_new_parameters_stay_out_of_the_param_lr_group():
    """Recorded as a deliberate choice, not an oversight, exactly as GeoFiLM's `value_film` was:
    these tensors are zero-init, but adding them to IDENTITY_INIT_PARAM_SUFFIXES would change what
    --param-lr-mult means for every existing cell. The queued cells leave it at its default 1.0."""
    m = _small_model(2, pose_reinject=True, geo_ray_gain=True)
    hit = sorted(n for n, _ in m.named_parameters() if n.endswith(IDENTITY_INIT_PARAM_SUFFIXES))
    note("g4b_param_lr_group", hit)
    assert not any(("pose_rein" in n) or ("ray_gain" in n) for n in hit), hit


# ============================================================ G5 keys / checkpoint compatibility
def test_g5_only_the_declared_keys_are_added_at_the_real_width():
    for tag, kw, keys in ON:
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
    note("g5_new_state_dict_keys", {t: k for t, _, k in ON})


def test_g5b_existing_hop16_checkpoint_strict_loads_into_the_default_off_model():
    """The default-off contract that matters operationally: the real trained control checkpoint
    still loads with strict=True, and warm-starting a variant leaves exactly the new keys missing."""
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
    note("g5b_hop16_ckpt_strict_load", f"{os.path.basename(CKPT_DIR)} loaded with strict=True "
                                       f"({len(sd)} tensors)")
    for tag, kw, keys in ON:
        torch.manual_seed(0)
        mv = OAAv2Depth(C=256, nviews=2, stem_stride1=True, **kw)
        out = mv.load_state_dict(sd, strict=False)
        assert sorted(out.missing_keys) == keys, (tag, out.missing_keys)
        assert list(out.unexpected_keys) == [], out.unexpected_keys
    note("g5b_warm_start_missing_keys", "exactly the new tensors for all 3 switch settings")


def test_g5c_config_round_trips_through_core_ckpt():
    """Save {state_dict, args} -> core.ckpt.build(args, DM) -> strict load -> identical prediction.
    Without this a cell trains for 2 hours and is then UNSCOREABLE: eval.py catches the strict-load
    failure per run and prints only `[skip <run>]`, so the OAA row vanishes from the table and the
    baselines look like the whole comparison. That is the hole `11b7202` had to patch for GeoFiLM."""
    from core.ckpt import build
    DM = types.SimpleNamespace(IN_CH={"r2": 2, "cB": 4, "fb": 4, "r6": 6, "r8": 8}, POSES={},
                               WINDOW=0, HOP=160, WIN=400, N_FFT=512)
    DM.__name__ = "data_0422"
    cfgs = [{}, {"pose_reinject": True}, {"geo_ray_gain": True},
            {"pose_reinject": True, "geo_ray_gain": True},
            {"geo_ray_gain": True, "geo_basis": "sh4"},
            {"geo_ray_gain": True, "geo_value_film": True},
            {"pose_reinject": True, "qk_norm": True, "registers": 4}]
    for extra in cfgs:
        args = dict(nviews=2, data_mode="r2", dim=64, rounds=2, lift_h=4, lift_w=8,
                    stem_stride1=True, max_depth=10.0, dec_deep=True, rounds_wired=True,
                    cond_mode="adaln", full_res_enc=True, multi_scale_lift=True,
                    data_module="data_0422", **extra)
        torch.manual_seed(0)
        src = OAAv2Depth(C=64, nviews=2, lh=4, lw=8, stem_stride1=True,
                         qk_norm=extra.get("qk_norm", False),
                         registers=extra.get("registers", 0),
                         geo_basis=extra.get("geo_basis", "raw"),
                         geo_value_film=extra.get("geo_value_film", False),
                         pose_reinject=extra.get("pose_reinject", False),
                         geo_ray_gain=extra.get("geo_ray_gain", False)).eval()
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


# ============================================================ G6 not inert once trained
def test_g6_real_weights_move_the_output_and_rezeroing_restores_it_exactly():
    """Otherwise G2 proves nothing: each switch has to be capable of changing the prediction, and
    putting its parameters back at the init has to restore the reference EXACTLY."""
    moved, restored = {}, 0.0
    for tag, kw, keys in ON:
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
def test_g7_pose_reinject_actually_differs_per_observation():
    """The mechanism claim for variant 1, checked on tensors rather than argued: the term added
    before each inter block must differ ACROSS OBSERVATIONS (otherwise it is a constant offset and
    could not help the inter-observation comparison), and each round's gamma must be its own
    tensor."""
    nv = 8
    m = _small_model(nv, pose_reinject=True)
    _set_nonzero(m, std=0.3)
    pose_feat, poses = m._pose_tensors(torch.device("cpu"))
    pc = m.pose_emb(pose_feat)                                           # (nv,C)
    spreads = {}
    for i in range(len(m.pose_rein)):
        t = (m.pose_rein[i] * pc)
        spreads[f"round{i}"] = round(float(t.std(0).max()), 5)
        assert float(t.std(0).max()) > 1e-6, i
    # the two rounds must not be the same tensor object or the same values
    assert m.pose_rein[0] is not m.pose_rein[1]
    assert float((m.pose_rein[0] - m.pose_rein[1]).abs().max()) > 0.0
    note("g7_pose_term_spread_over_observations", spreads)
    # and at r2, where both observations share yaw 0 and differ only in ear sign, the term must
    # still separate the two ears -- that is the whole binaural content of the pose at 2 observations
    m2 = _small_model(2, pose_reinject=True)
    _set_nonzero(m2, std=0.3)
    pf2, _ = m2._pose_tensors(torch.device("cpu"))
    pc2 = m2.pose_emb(pf2)
    sep = float((m2.pose_rein[0] * pc2)[0].sub((m2.pose_rein[0] * pc2)[1]).abs().max())
    note("g7_pose_term_r2_ear_separation", round(sep, 5))
    assert sep > 1e-6, sep


def test_g7b_ray_gain_is_per_ray_per_head_and_constant_over_observations():
    """The mechanism claim for variant 2, and the line that separates it from GeoFiLM. The gain must
    vary over rays and over heads, and it must have NO observation axis at all: GeoFiLM reweights
    observations against each other, this one can change only the total a ray reads out. Also
    checked: --dump-attn-stats reports it, and at init every reported number is exactly 1.0 / 0.0."""
    rm = _ray_mod(geo_ray_gain=True)
    with torch.no_grad():
        torch.nn.init.normal_(rm.ray_gain.weight, std=0.3)
        torch.nn.init.normal_(rm.ray_gain.bias, std=0.1)
    rm.collect_stats = True
    q, tk, d3, poses = _ray_inputs(2, 16, 8, 8, 64)
    rm(q, tk, d3, poses, 8)
    feats = rm._geo_basis_feats(d3, poses)                               # (R,N,gf)
    rf = torch.cat([feats.mean(1), d3.to(feats.dtype)], -1)              # (R,gf+3)
    rg = 1.0 + rm.ray_gain(rf)                                           # (R,h)
    assert rg.shape == (16, 8), rg.shape
    spread = {"over_rays": round(float(rg.std(0).max()), 4),
              "over_heads": round(float(rg.std(1).max()), 4)}
    note("g7b_ray_gain_spread", spread)
    for k, v in spread.items():
        assert v > 1e-6, (k, v)
    note("g7b_ray_gain_has_no_observation_axis", f"shape {tuple(rg.shape)} = (R, heads)")
    for k in ("ray_gain_mean", "ray_gain_std", "ray_gain_min", "ray_gain_max"):
        assert k in rm.stats, k
    assert rm.stats["ray_gain_mean_per_head"].shape == (8,)
    assert rm.stats["ray_gain_per_ray"].shape == (16,)
    note("g7b_ray_gain_stats_trained",
         {k: round(rm.stats[k], 4) for k in ("ray_gain_mean", "ray_gain_std",
                                             "ray_gain_min", "ray_gain_max")})
    rm0 = _ray_mod(geo_ray_gain=True)
    rm0.collect_stats = True
    rm0(q, tk, d3, poses, 8)
    assert rm0.stats["ray_gain_mean"] == 1.0 and rm0.stats["ray_gain_min"] == 1.0, rm0.stats
    assert rm0.stats["ray_gain_max"] == 1.0 and rm0.stats["ray_gain_std"] == 0.0, rm0.stats
    note("g7b_ray_gain_stats_at_init", "mean/min/max = 1.0 exactly, std = 0.0 exactly")


def test_g7c_the_observation_mean_geometry_column_collapses_for_a_symmetric_rig():
    """An adverse finding recorded as a test so it cannot quietly go away: for the fixed rigs we
    train on, the observation-mean geometry feature loses most of its information (at r8 the mean
    rotation matrix is exactly zero and the ear columns cancel pairwise), which is WHY the raw ray
    direction is concatenated. Without those 3 columns the gain would have one live input at r8."""
    rows = {}
    for nv in (2, 4, 8):
        m = _small_model(nv, geo_ray_gain=True)
        pose_feat, poses = m._pose_tensors(torch.device("cpu"))
        d3 = m.dir6_buf[:, :3].double()
        feats = m.ray_mic._geo_basis_feats(d3, poses)                    # (R,N,5)
        mn = feats.mean(1)                                               # (R,5)
        live = [i for i in range(mn.shape[-1]) if float(mn[:, i].abs().max()) > 1e-12]
        rows[nv] = {"live_columns_of_5": live,
                    "colwise_absmax": [round(float(mn[:, i].abs().max()), 4)
                                       for i in range(mn.shape[-1])]}
    assert rows[8]["live_columns_of_5"] == [1], rows[8]      # dy only
    note("g7c_observation_mean_geometry_live_columns", rows)


def test_g7d_obs_mask_invariance_survives_trained_parameters():
    """With real (non-identity) parameters an invalid observation still must not change the
    prediction. The gain reads geometry only and its observation mean is deliberately UNMASKED, so
    it does not depend on obs_valid at all; the pose term is deterministic per observation and a
    dropped observation is removed by the existing masked mean. This is the invariance
    `--subset-aug` depends on."""
    nv = 8
    rows = {}
    for tag, kw, _ in ON:
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
    note("g7d_mask_invariance_with_trained_params", rows)


# ============================================================ refusals
def test_incoherent_combinations_are_refused():
    """--pose-reinject re-states the pose code --no-pose-emb removes, so the two are the same
    pathway in opposite directions. --geo-ray-gain reads the geometry features, which --no-geo-bias
    never computes, and needs a cross-attention output, which --no-cross deletes."""
    cases = [("pose+no_pose_emb", dict(pose_reinject=True, no_pose_emb=True), "same pathway"),
             ("gain+no_geo_bias", dict(geo_ray_gain=True, no_geo_bias=True), "geo-bias"),
             ("gain+no_cross", dict(geo_ray_gain=True, no_cross=True), "no-cross")]
    for tag, kw, want in cases:
        try:
            _small_model(4, **kw)
            raise RuntimeError(f"{tag} should have been refused")
        except AssertionError as e:
            assert want in str(e), (tag, str(e))
    # --pose-reinject lives in OAAv2Depth, not RayMicAttn, so it is legitimate with --no-cross
    _small_model(4, pose_reinject=True, no_cross=True)
    note("refusals", "pose+no-pose-emb / gain+no-geo-bias / gain+no-cross raise; "
                     "pose+no-cross is allowed on purpose")


# ============================================================ run-level contracts
def test_forward_contract_for_the_queued_cell_configs():
    """The exact configurations the cells would queue (`--stem-stride1` plus one switch): output
    (B,1,256,512) in [0,1], finite, and a live finite gradient on every new parameter."""
    for tag, kw, keys in ON:
        for nv in (2, 4, 8):
            m = _small_model(nv, stem_stride1=True, **kw).train()
            D = m(_spec(2, nv))
            assert D.shape == (2, 1, 256, 512) and torch.isfinite(D).all(), (tag, nv)
            assert float(D.min()) >= 0.0 and float(D.max()) <= 1.0, (tag, nv)
            D.sum().backward()
            pm = dict(m.named_parameters())
            for q in keys + ["ray_mic.q.weight", "ray_mic.bias_mlp.0.weight", "pose_emb.0.weight"]:
                g = pm[q].grad
                assert g is not None and torch.isfinite(g).all(), (tag, nv, q)
                assert float(g.abs().sum()) > 0, (tag, nv, q)
    note("cell_forward_contract",
         "3 switch settings x N=2/4/8 with --stem-stride1: (2,1,256,512) in [0,1], live gradients")


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
