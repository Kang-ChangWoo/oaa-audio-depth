"""Contract tests for the OAA attention/fusion variants (qk-norm, ray-gate, view-fusion, obs mask).

Run either way:
    python tests/test_oaa_attention_variants.py        # prints every measured number
    pytest -q tests/test_oaa_attention_variants.py

Everything that is an *equivalence* claim runs on CPU in float64. That is deliberate: on GPU the
TF32/flash-attention kernels are not bit-reproducible, so a CUDA run can only ever establish a loose
bound, while float64 on CPU separates "mathematically identical" from "numerically close". The
GPU-only checks (bf16 autocast, peak VRAM, step time, real checkpoints) live in
`tests/gpu_smoke_attn_fusion.py` and are reported separately.

`OAA_REF_MODULE` points at a pristine pre-change copy of model/oaa.py for the output-parity test;
default `/root/local1/changwoo/attn_fusion_20261013/ref/oaa_ref.py`. Missing -> that one test skips
with its reason recorded.
"""
import importlib.util
import math
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import oaa as M                                      # noqa: E402
from model.oaa import OAAv2Depth, RayMicAttn                    # noqa: E402

REF_PATH = os.environ.get("OAA_REF_MODULE", "/root/local1/changwoo/attn_fusion_20261013/ref/oaa_ref.py")
SMALL = dict(C=64, lh=4, lw=8, enc_res=(16, 32))                 # 256x512 output, tiny attention
NOTES = []                                                      # (name, measured number) for the report


def note(k, v):
    NOTES.append((k, v))
    print(f"    {k} = {v}")


# --------------------------------------------------------------------------- helpers
def _ray_mod(N, dk=8, heads=8, **kw):
    torch.manual_seed(0)
    return RayMicAttn(dk * heads, heads=heads, **kw).double().eval()


def _copy_shared(src, dst):
    """Copy every parameter/buffer the two modules have in common (new options add keys only)."""
    s, d = src.state_dict(), dst.state_dict()
    missing = dst.load_state_dict({k: v for k, v in s.items() if k in d}, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    return sorted(set(d) - set(s))                              # the keys the variant added


def _ray_inputs(B, R, N, Mtok, C, seed=1, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    q_in = torch.randn(B, R, C, generator=g, dtype=torch.float64) * scale
    tokens = torch.randn(B, N * Mtok, C, generator=g, dtype=torch.float64) * scale
    el = (torch.arange(R, dtype=torch.float64) + 0.5) / R * math.pi - math.pi / 2
    dir3 = torch.stack([torch.cos(el), torch.sin(el), torch.cos(el) * 0.5], -1)
    poses = [(2 * math.pi * i / max(N // 2, 1), (-1.0, 1.0)[i % 2]) for i in range(N)]
    return q_in, tokens, dir3, poses


def _small_model(nviews, **kw):
    torch.manual_seed(0)
    return OAAv2Depth(nviews=nviews, **SMALL, **kw).double().eval()


def _spec(B, nviews, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, nviews, 24, 40, generator=g, dtype=torch.float64)


def _maxdiff(a, b):
    return float((a - b).abs().max())


# --------------------------------------------------------------------------- 9.1 basic contract
def test_forward_contract_all_nviews():
    """(B,1,256,512) in [0,1], finite, for N = 2,4,6,8; gradients reach every new parameter."""
    for nv in (2, 4, 6, 8):
        for kw in ({}, dict(qk_norm=True), dict(ray_gate=True),
                   dict(view_fusion="hier"), dict(view_fusion="hier-geo"),
                   dict(qk_norm=True, ray_gate=True, view_fusion="hier-geo")):
            m = _small_model(nv, **kw).train()
            x = _spec(2, nv).requires_grad_(True)
            D = m(x)
            assert D.shape == (2, 1, 256, 512), (nv, kw, D.shape)
            assert torch.isfinite(D).all(), (nv, kw)
            assert float(D.min()) >= 0.0 and float(D.max()) <= 1.0, (nv, kw)
            D.sum().backward()
            for nm in ("ray_mic.q", "ray_mic.k", "ray_mic.v", "ray_mic.bias_mlp.0"):
                g = dict(m.named_parameters())[nm + ".weight"].grad
                assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0, (nv, kw, nm)
            for nm in ("ray_mic.logit_scale", "ray_mic.geo_scale", "ray_mic.gate_proj.weight",
                       "ray_mic.route_mlp.2.weight"):
                p = dict(m.named_parameters()).get(nm)
                if p is not None:
                    assert p.grad is not None and torch.isfinite(p.grad).all(), (nv, kw, nm)
    note("forward_contract_configs", f"{4 * 6} (4 nviews x 6 variant combos) all (2,1,256,512) in [0,1]")


def test_view_poses_override_and_cue_ablations():
    """A view_poses override and the pose/ear/yaw cue ablations still run and still change the output."""
    nv = 4
    m = _small_model(nv)
    x = _spec(2, nv)
    base = m(x)
    flip = m(x, view_poses=[(-y, e) for (y, e) in m.view_pose])
    assert torch.isfinite(flip).all()
    note("yawflip_override_maxdiff", f"{_maxdiff(base, flip):.3e} (must be > 0: poses do reach the model)")
    assert _maxdiff(base, flip) > 1e-9
    for kw in (dict(no_pose_emb=True), dict(no_ray_emb=True), dict(no_geo_bias=True), dict(no_tf_pe=True),
               dict(no_cross=True)):
        mm = _small_model(nv, **kw)
        assert torch.isfinite(mm(x)).all(), kw


def test_param_count_default_equals_reference():
    """Default-off model must have exactly the reference parameter count; variants are measured."""
    base = _small_model(8)
    nb = sum(p.numel() for p in base.parameters())
    rows = {}
    for name, kw in (("E1 qk-norm", dict(qk_norm=True)), ("E2 ray-gate", dict(ray_gate=True)),
                     ("E4 hier", dict(view_fusion="hier")), ("E4 hier-geo", dict(view_fusion="hier-geo")),
                     ("E3 qk+gate", dict(qk_norm=True, ray_gate=True))):
        n = sum(p.numel() for p in _small_model(8, **kw).parameters())
        rows[name] = n - nb
    note("param_delta_small_cfg", rows)
    assert rows["E4 hier"] == 0, "hier must add no parameters"
    assert rows["E1 qk-norm"] == 16, "two per-head log-scales at 8 heads"
    assert rows["E2 ray-gate"] == 64 * 8 + 8, "gate_proj C->heads with bias"
    assert rows["E3 qk+gate"] == rows["E1 qk-norm"] + rows["E2 ray-gate"]


# --------------------------------------------------------------------------- 9.2.1/2 parity
def test_default_state_dict_strict_loads_reference_and_matches_output():
    """The pre-change module and the new module, default options, must agree bit-for-bit in float64
    and exchange state dicts with strict=True."""
    if not os.path.exists(REF_PATH):
        note("baseline_parity", f"SKIPPED: no reference module at {REF_PATH}")
        return
    spec = importlib.util.spec_from_file_location("oaa_ref", REF_PATH)
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    # float32, and the bar is max|diff| == 0.0 exactly. Parity is a claim about running the same
    # ops in the same order, not about precision, so float32 is the right place to demand bitwise
    # equality -- and the pristine reference predates the dtype-polymorphism fix, so it cannot run
    # in float64 at all (torch.tensor() without dtype= in _pose_tensors / the ear-axis vector).
    for nv in (2, 8):
        torch.manual_seed(0); old = ref.OAAv2Depth(nviews=nv, **SMALL).eval()
        torch.manual_seed(0); new = OAAv2Depth(nviews=nv, **SMALL).eval()
        assert set(old.state_dict()) == set(new.state_dict()), "default key sets differ"
        new.load_state_dict(old.state_dict(), strict=True)       # the strict-load contract
        x = _spec(2, nv).float()
        with torch.no_grad():
            d = _maxdiff(old(x), new(x))
        note(f"baseline_parity_N{nv}_maxdiff_fp32", f"{d:.3e}")
        assert d == 0.0, f"default path is not bit-identical to the reference (N={nv}, {d:.3e})"
        assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    # and the reference's own RayMicAttn must match the new one on the default path, incl. grads
    for nv in (2, 4, 6, 8):
        torch.manual_seed(0); ro = ref.RayMicAttn(64, heads=8).eval()
        torch.manual_seed(0); no = RayMicAttn(64, heads=8).eval()
        no.load_state_dict(ro.state_dict(), strict=True)
        q, tk, d3, poses = _ray_inputs(2, 16, nv, 8, 64)
        outs, grads = [], []
        for mod in (ro, no):
            mod.zero_grad()
            qq, tt = q.float().clone().requires_grad_(True), tk.float().clone().requires_grad_(True)
            o = mod(qq, tt, d3.float(), poses, 8)
            o.square().sum().backward()
            outs.append(o.detach()); grads.append([qq.grad, tt.grad, mod.bias_mlp[0].weight.grad])
        assert _maxdiff(*outs) == 0.0, nv
        assert max(_maxdiff(a, b) for a, b in zip(*grads)) == 0.0, nv
    note("baseline_raymicattn_parity", "N=2,4,6,8: forward and Q/K/V/bias_mlp grads max|diff| = 0.0")


# --------------------------------------------------------------------------- 9.2.3 E2
def test_gate_init_is_identity():
    """2*sigmoid(0) = 1, so a freshly built ray_gate module must match the ungated one exactly."""
    for N in (2, 8):
        off = _ray_mod(N)
        on = _ray_mod(N, ray_gate=True)
        added = _copy_shared(off, on)
        assert added == ["gate_proj.bias", "gate_proj.weight"], added
        args = _ray_inputs(2, 16, N, 8, 64)
        d = _maxdiff(off(*args, 8), on(*args, 8))
        note(f"gate_init_identity_N{N}_maxdiff", f"{d:.3e}")
        assert d < 1e-12, d
    # and the gate is not inert once trained: a non-zero gate_proj must change the output
    on = _ray_mod(8, ray_gate=True)
    torch.nn.init.normal_(on.gate_proj.weight, std=0.1)
    args = _ray_inputs(2, 16, 8, 8, 64)
    assert _maxdiff(_ray_mod(8)(*args, 8), on(*args, 8)) > 1e-6


# --------------------------------------------------------------------------- 9.2.4 E4 exactness
def test_hier_equals_global_forward_and_grad():
    """hier is an exact decomposition of the one global softmax: same output AND same gradients
    w.r.t. Q/K/V projections and bias_mlp, over N = 2,4,6,8, geometry bias on and off, and
    deliberately non-uniform logit scales."""
    worst_out = worst_grad = 0.0
    worst_where = None
    for N in (2, 4, 6, 8):
        for use_bias in (True, False):
            for scale in (0.5, 3.0):                             # 3.0 makes the softmax sharp
                g_mod = _ray_mod(N, use_bias=use_bias, view_fusion="global")
                h_mod = _ray_mod(N, use_bias=use_bias, view_fusion="hier")
                assert _copy_shared(g_mod, h_mod) == [], "hier must add no state"
                q, tk, d3, poses = _ray_inputs(2, 16, N, 8, 64, scale=scale)
                outs, grads = [], []
                for mod in (g_mod, h_mod):
                    mod.zero_grad()
                    qq, tt = q.clone().requires_grad_(True), tk.clone().requires_grad_(True)
                    o = mod(qq, tt, d3, poses, 8)
                    (o * torch.linspace(0.3, 1.7, o.numel(), dtype=torch.float64).view_as(o)).sum().backward()
                    outs.append(o.detach())
                    named = [("input.q", qq.grad), ("input.tokens", tt.grad)]
                    named += [(n, p.grad) for n, p in sorted(mod.named_parameters(), key=lambda t: t[0])
                              if p.grad is not None]
                    grads.append(named)
                do = _maxdiff(*outs)
                # Scale the gradient comparison by the LARGEST gradient in the config, not by each
                # tensor's own max: several parameters have near-zero gradients here, and dividing an
                # eps-level difference by its own ~1e-12 magnitude manufactures a 1e-4 "error".
                gscale = max(float(a.abs().max()) for _, a in grads[0])
                for (na, a), (nb, b) in zip(*grads):
                    assert na == nb
                    dg = _maxdiff(a, b) / gscale
                    if dg > worst_grad:
                        worst_grad, worst_where = dg, f"{na} @ N={N} bias={use_bias} scale={scale}"
                worst_out = max(worst_out, do)
                assert do < 1e-13, (N, use_bias, scale, do)
                assert worst_grad < 1e-12, (N, use_bias, scale, worst_grad, worst_where)
    note("hier_vs_global_out_maxdiff", f"{worst_out:.3e} (float64, worst of 16 configs; "
                                       f"double eps = 2.2e-16)")
    note("hier_vs_global_grad_maxrel", f"{worst_grad:.3e} (relative to the largest gradient in the "
                                       f"config; worst at {worst_where})")


def test_hier_is_not_a_mean_over_observations():
    """Guard the one wrong implementation the spec forbids: a plain mean across observations."""
    N, Mtok, C = 4, 8, 64
    h_mod = _ray_mod(N, view_fusion="hier")
    q, tk, d3, poses = _ray_inputs(2, 16, N, Mtok, C, scale=3.0)
    hier = h_mod(q, tk, d3, poses, Mtok)

    class MeanFuse(type(h_mod)):
        def _hier_fuse(self, logits, V, N_, M_, feats, kv_keep):
            L = logits.reshape(*logits.shape[:3], N_, M_).double()
            z = torch.einsum("bhrnm,bhnmd->bhrnd", L.softmax(-1), V.reshape(*V.shape[:2], N_, M_, self.dk).double())
            return z.mean(-2).to(V.dtype)
    bad = MeanFuse(C, heads=8, view_fusion="hier").double().eval()
    _copy_shared(h_mod, bad)
    d = _maxdiff(hier, bad(q, tk, d3, poses, Mtok))
    note("hier_vs_uniform_mean_maxdiff", f"{d:.3e} (must be large: the decomposition is NOT a mean)")
    assert d > 1e-3


# --------------------------------------------------------------------------- 9.2.5 E4 router init
def test_hier_geo_init_equals_hier():
    """Zero-initialised routing residual => hier-geo starts exactly at hier."""
    for N in (2, 8):
        h_mod = _ray_mod(N, view_fusion="hier")
        hg = _ray_mod(N, view_fusion="hier-geo")
        added = _copy_shared(h_mod, hg)
        assert all(k.startswith("route_mlp.") for k in added), added
        args = _ray_inputs(2, 16, N, 8, 64)
        d = _maxdiff(h_mod(*args, 8), hg(*args, 8))
        note(f"hier_geo_init_identity_N{N}_maxdiff", f"{d:.3e}")
        assert d < 1e-12, d
    hg = _ray_mod(8, view_fusion="hier-geo")
    torch.nn.init.normal_(hg.route_mlp[-1].weight, std=0.5)
    args = _ray_inputs(2, 16, 8, 8, 64)
    assert _maxdiff(_ray_mod(8, view_fusion="hier")(*args, 8), hg(*args, 8)) > 1e-6


# --------------------------------------------------------------------------- 9.2.6-8 E6 masking
def test_all_valid_mask_equals_no_mask():
    """obs_valid all-True must equal obs_valid=None. The production short-circuit makes this exact,
    so the masked code path is forced on to make the check meaningful."""
    try:
        M._SHORTCIRCUIT_ALL_VALID = False
        for vf in ("global", "hier", "hier-geo"):
            for nv in (2, 8):
                m = _small_model(nv, view_fusion=vf)
                x = _spec(2, nv)
                d = _maxdiff(m(x), m(x, obs_valid=torch.ones(2, nv, dtype=torch.bool)))
                note(f"allvalid_identity_{vf}_N{nv}_maxdiff", f"{d:.3e}")
                assert d < 1e-10, (vf, nv, d)
    finally:
        M._SHORTCIRCUIT_ALL_VALID = True


def test_invalid_observations_cannot_change_the_output():
    """The whole point of E6: perturb a masked-out observation's spectrogram arbitrarily and the
    prediction must not move. A zeroed spectrogram alone does NOT give this."""
    for vf in ("global", "hier", "hier-geo"):
        nv = 8
        m = _small_model(nv, view_fusion=vf)
        ov = torch.ones(2, nv, dtype=torch.bool); ov[:, [1, 4, 6]] = False
        x = _spec(2, nv)
        a = m(x, obs_valid=ov)
        x2 = x.clone()
        x2[:, [1, 4, 6]] = torch.rand_like(x2[:, [1, 4, 6]]) * 7.0 + 3.0     # wildly different values
        b = m(x2, obs_valid=ov)
        d = _maxdiff(a, b)
        note(f"invalid_invariance_{vf}_maxdiff", f"{d:.3e}")
        assert d < 1e-12, (vf, d)
        # control: the same perturbation WITHOUT the mask does move the output
        dz = _maxdiff(m(x), m(x2))
        note(f"invalid_invariance_{vf}_unmasked_control", f"{dz:.3e} (must be large)")
        assert dz > 1e-6, vf


def test_zeroed_spectrogram_is_not_the_same_as_masking():
    """Legacy --subset-aug zeroing vs explicit masking must give different predictions; if they did
    not, E6 would have nothing to test."""
    nv = 8
    m = _small_model(nv)
    x = _spec(2, nv); x[:, [1, 4]] = 0
    ov = torch.ones(2, nv, dtype=torch.bool); ov[:, [1, 4]] = False
    d = _maxdiff(m(x), m(x, obs_valid=ov))
    note("zeroonly_vs_truemask_maxdiff", f"{d:.3e} (must be > 0)")
    assert d > 1e-6


def test_extreme_mask_has_no_nan():
    """8 observations, exactly one valid, every fusion mode; and the guard against zero valid."""
    nv = 8
    for vf in ("global", "hier", "hier-geo"):
        m = _small_model(nv, view_fusion=vf)
        ov = torch.zeros(2, nv, dtype=torch.bool); ov[:, 3] = True
        D = m(_spec(2, nv), obs_valid=ov)
        assert torch.isfinite(D).all(), vf
        note(f"single_valid_obs_{vf}", f"finite, range [{float(D.min()):.4f}, {float(D.max()):.4f}]")
    m = _small_model(nv)
    try:
        m(_spec(2, nv), obs_valid=torch.zeros(2, nv, dtype=torch.bool))
        raise AssertionError("all-invalid input must be refused, not silently NaN")
    except AssertionError as e:
        assert "at least one valid observation" in str(e), e
    # and a dropped observation really carries zero attention mass
    rm = _ray_mod(8, view_fusion="hier"); rm.collect_stats = True
    q, tk, d3, poses = _ray_inputs(2, 16, 8, 8, 64)
    ov = torch.ones(2, 8, dtype=torch.bool); ov[:, [0, 5]] = False
    rm(q, tk, d3, poses, 8, ov)
    am = rm.stats["alpha_mean_per_view"]                         # (N,)
    hv = rm.stats["alpha_mean_head_view"]                        # (heads, N)
    note("hier_alpha_of_dropped_views", f"views 0,5 -> {[float(v) for v in am[[0, 5]]]} (exactly 0); "
                                        f"kept views sum to {float(am.sum()):.6f}")
    assert float(am[[0, 5]].abs().max()) == 0.0
    assert float(hv[:, [0, 5]].abs().max()) == 0.0, "no head may put mass on a dropped observation"
    assert abs(float(am.sum()) - 1.0) < 1e-9, "alpha must still be a distribution over the kept views"


# --------------------------------------------------------------------------- 9.2.9 E5 loss
def test_spherical_loss_analytic():
    """Analytic cases for the latitude-weighted per-image L1, plus the empty-mask contract."""
    from core.metrics import cos_lat, depth_loss
    H, W = 256, 512
    wlat = cos_lat(H, torch.device("cpu")).view(1, 1, H, 1)
    gt = torch.full((2, 1, H, W), 0.4)
    mask = torch.ones_like(gt)
    D = gt + 0.07                                                # constant error everywhere
    v = float(depth_loss(D, gt, mask, "spherical", wlat))
    note("spherical_constant_error", f"{v:.8f} (expected 0.07000000)")
    assert abs(v - 0.07) < 1e-6
    assert abs(float(depth_loss(D, gt, mask, "plain", wlat)) - 0.07) < 1e-6
    # error on one latitude row only -> per-image weighted mean = e * w_row / sum_rows(w)
    D2 = gt.clone(); D2[:, :, 100, :] += 0.5
    exp = 0.5 * float(wlat[0, 0, 100, 0]) / float(wlat[0, 0, :, 0].sum())
    v = float(depth_loss(D2, gt, mask, "spherical", wlat))
    note("spherical_single_row_error", f"{v:.8f} (expected {exp:.8f})")
    assert abs(v - exp) < 1e-6
    # the plain loss weights that row by pixel area instead, so the two must disagree
    vp = float(depth_loss(D2, gt, mask, "plain", wlat))
    note("plain_single_row_error", f"{vp:.8f} (unweighted 0.5/256 = {0.5 / 256:.8f})")
    assert abs(vp - 0.5 / 256) < 1e-6
    assert abs(v - vp) > 1e-5, "spherical and plain must differ off the equator"
    # half the batch unusable -> only the usable images count; fully empty -> None, never NaN
    half = mask.clone(); half[1] = 0
    vh = float(depth_loss(D, gt, half, "spherical", wlat))
    note("spherical_one_empty_image", f"{vh:.8f} (expected 0.07, the empty image is dropped)")
    assert abs(vh - 0.07) < 1e-6
    assert depth_loss(D, gt, torch.zeros_like(mask), "spherical", wlat) is None, "empty mask -> None"


# --------------------------------------------------------------------------- 9.2.10 round trip
def test_checkpoint_round_trip_through_core_ckpt():
    """Save {state_dict, args} -> core.ckpt.build(args, DM) -> strict load -> identical prediction,
    for every representative new configuration AND for a default (old-style) args dict."""
    from core.ckpt import build
    DM = types.SimpleNamespace(IN_CH={"r2": 2, "cB": 4, "r6": 6, "r8": 8}, POSES={}, WINDOW=0,
                               HOP=160, WIN=400, N_FFT=512)
    DM.__name__ = "data_0422"
    cfgs = [{}, {"qk_norm": True}, {"ray_gate": True}, {"view_fusion": "hier"},
            {"view_fusion": "hier-geo"}, {"qk_norm": True, "ray_gate": True, "view_fusion": "hier-geo"}]
    preds = 0
    for i, extra in enumerate(cfgs):
        # dim 64 / lift 4x8 keeps the CPU forward affordable; the rebuild contract is independent of width
        args = dict(nviews=8, data_mode="r8", dim=64, rounds=2, lift_h=4, lift_w=8,
                    stem_stride1=True, max_depth=10.0, dec_deep=True, rounds_wired=True,
                    cond_mode="adaln", full_res_enc=True, multi_scale_lift=True,
                    data_module="data_0422", **extra)
        torch.manual_seed(0)
        src = OAAv2Depth(C=64, nviews=8, lh=4, lw=8, stem_stride1=True,
                         qk_norm=extra.get("qk_norm", False), ray_gate=extra.get("ray_gate", False),
                         view_fusion=extra.get("view_fusion", "global")).eval()
        rebuilt, dmode, nch, kind, poses = build(args, DM)
        assert set(rebuilt.state_dict()) == set(src.state_dict()), extra
        rebuilt.load_state_dict(src.state_dict(), strict=True)   # the contract under test
        rebuilt.eval()
        assert (dmode, nch, kind) == ("r8", 8, "spec")
        if i < 3 or i == len(cfgs) - 1:                          # predictions on 4 of the 6
            x = torch.rand(1, 8, 24, 40)
            with torch.no_grad():
                d = _maxdiff(src(x), rebuilt(x))
            assert d == 0.0, (extra, d)
            preds += 1
    note("ckpt_round_trip", f"{len(cfgs)} configs strict-loaded, {preds} of them also "
                            f"prediction-compared: max|diff| = 0.0")
    # an args dict with NO variant keys (every pre-2026-10-13 checkpoint) must rebuild the reference
    old = dict(nviews=8, data_mode="r8", rounds_wired=True, cond_mode="adaln", full_res_enc=True,
               multi_scale_lift=True)
    m, *_ = build(old, DM)
    assert (m.qk_norm, m.ray_gate, m.view_fusion) == (False, False, "global")
    torch.manual_seed(0)
    assert set(m.state_dict()) == set(OAAv2Depth(C=256, nviews=8).state_dict())
    note("old_args_rebuild", "no variant keys -> global/off, identical state_dict key set")


def test_no_cross_refuses_the_variants():
    """--no-cross deletes RayMicAttn, so the variants must be refused rather than silently ignored."""
    for kw in (dict(qk_norm=True), dict(ray_gate=True), dict(view_fusion="hier-geo")):
        try:
            _small_model(4, no_cross=True, **kw)
            raise RuntimeError(f"no_cross + {kw} should have been refused")
        except AssertionError as e:
            assert "no-cross" in str(e), e
    note("no_cross_conflict", "qk-norm / ray-gate / view-fusion all refused with --no-cross")


# --------------------------------------------------------------------------- runner
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        print(f"[{t.__name__}]")
        try:
            t()
            print("  PASS")
        except Exception as e:                                   # noqa: BLE001 - report, do not hide
            failed.append((t.__name__, repr(e)))
            print(f"  FAIL {e!r}")
    print("\n==== measured numbers ====")
    for k, v in NOTES:
        print(f"{k}: {v}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    for n, e in failed:
        print(f"FAILED {n}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
