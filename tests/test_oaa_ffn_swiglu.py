"""Two-part gate for `--ffn-swiglu {off,wide,matched}` (D1 / D2, 2026-10-16).

Run either way:
    python tests/test_oaa_ffn_swiglu.py        # prints every measured number
    pytest -q tests/test_oaa_ffn_swiglu.py

**Why this gate has two parts and not one.** Every archfix switch, and GeoFiLM, had a zero or
identity initialisation that recovered the reference bit for bit, so a single `max|diff| == 0.0`
gate was enough. SwiGLU does not: it replaces `W2 GELU(W1 h)` with `down(silu(gate(h)) * up(h))`,
and no setting of the new weights turns a multiplicative nonlinearity back into GELU. Claiming
bit-identity here would be claiming something impossible. So, exactly as archfix B1 did for
`--fine-layerscale`:

    (i)  the flag OFF is bit-identical to the parent commit and still strict-loads old
         checkpoints, CPU float64, N = 2/4/8;
    (ii) the flag ON has a sane forward and a LIVE gradient on every tensor -- the set of
         zero-gradient parameters is enumerated and reported, not asserted away -- and the
         parameter count is measured, not estimated.

`OAA_REF_MODULE_11B7202` points at a pristine `11b7202` copy of model/oaa.py (default
`/root/local1/changwoo/archfix_20261016/ref/oaa_ref_11b7202.py`, produced with
`git show 11b7202:model/oaa.py`). `11b7202` is this change's parent. Missing -> the parity tests
FAIL rather than silently pass, unless OAA_REF_OPTIONAL=1 records an explicit, counted skip.

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
from model.oaa import (OAAv2Depth, SelfAttn, SWIGLU_MODES,              # noqa: E402
                       _SwiGLUFFN, _swiglu_hidden)

REF_PATH = os.environ.get("OAA_REF_MODULE_11B7202",
                          "/root/local1/changwoo/archfix_20261016/ref/oaa_ref_11b7202.py")
CKPT_DIR = os.environ.get(
    "ARCHFIX_CKPT",
    "/root/local1/changwoo/relevel_hop16_obs48_20261015/runs/rep/relevel_h16_stem1_r2_rep_s0")
SMALL = dict(C=64, lh=4, lw=8, enc_res=(16, 32))     # 256x512 output, tiny attention
CEILING_M = 15.45          # the archfix ceiling. D1 is ALLOWED to break it (Changwoo 2026-10-16).
N_ERP_BLOCKS = 4           # the scope: the 4 query-refinement blocks, same as the FFN-4x proposal
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
        msg = (f"pristine 11b7202 module not found at {REF_PATH}.\n"
               f"    git show 11b7202:model/oaa.py > {REF_PATH}\n"
               f"    or set OAA_REF_OPTIONAL=1 for an explicit, counted skip.")
        if os.environ.get("OAA_REF_OPTIONAL") == "1":
            raise _Skipped(msg)
        raise FileNotFoundError(msg)
    spec = importlib.util.spec_from_file_location("oaa_ref_11b7202", REF_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _small_model(nviews, ref=None, **kw):
    torch.manual_seed(0)
    cls = (ref or M).OAAv2Depth
    return cls(nviews=nviews, **SMALL, **kw).double().eval()


def _spec(B, nviews, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, nviews, 24, 40, generator=g, dtype=torch.float64)


def _copy_shared(src, dst):
    s, d = src.state_dict(), dst.state_dict()
    out = dst.load_state_dict({k: v for k, v in s.items() if k in d}, strict=False)
    assert not out.unexpected_keys, out.unexpected_keys
    return sorted(set(d) - set(s))


def _grads_of(m, x):
    m.zero_grad()
    D = m(x)
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


def _ffn_params(C, h):
    """LN(C) + Linear(C,h) gate + Linear(C,h) up + Linear(h,C) down."""
    return 2 * C + 2 * (C * h + h) + (h * C + C)


def _ref_ffn_params(C):
    """LN(C) + Linear(C,2C) + Linear(2C,C) = 4C^2 + 5C."""
    return 2 * C + (C * 2 * C + 2 * C) + (2 * C * C + C)


# ================================================== part (i): the flag OFF is bit-identical
def test_part_i_flag_off_is_bit_identical_to_11b7202():
    """The patch must be completely inert at the default: same forward, same gradients, same
    state_dict keys as the parent commit. Exactly 0.0, float64, N = 2/4/8."""
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
    note("i_off_vs_11b7202_forward_maxdiff", f"{worst_f:.3e} (N=2/4/8, must be exactly 0)")
    note("i_off_vs_11b7202_grad_maxrel", f"{worst_g:.3e} (must be exactly 0)")


def test_part_i_default_is_off_and_the_reference_sequential_survives():
    """The default must be the string "off", and at the default the block must still be the exact
    reference nn.Sequential with its m.0 / m.1 / m.3 keys -- not a SwiGLU that happens to look
    similar. An old checkpoint's keys depend on this."""
    assert SWIGLU_MODES[0] == "off", SWIGLU_MODES
    m = _small_model(2)
    assert m.ffn_swiglu == "off", m.ffn_swiglu
    for blk in m.erp:
        assert isinstance(blk.m, torch.nn.Sequential), type(blk.m)
        assert isinstance(blk.m[2], torch.nn.GELU), type(blk.m[2])
    keys = [k for k in m.state_dict() if k.startswith("erp.0.m.")]
    note("i_default_erp_ffn_keys", sorted(keys))
    assert sorted(keys) == ["erp.0.m.0.bias", "erp.0.m.0.weight", "erp.0.m.1.bias",
                            "erp.0.m.1.weight", "erp.0.m.3.bias", "erp.0.m.3.weight"], keys
    # and a SelfAttn built with the default keyword is the same thing
    torch.manual_seed(0)
    assert isinstance(SelfAttn(16).m, torch.nn.Sequential)


def test_part_i_existing_hop16_checkpoint_strict_loads_into_default_off_model():
    """The operational default-off contract: a real trained checkpoint from before this change
    still loads with strict=True."""
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
    OAAv2Depth(C=256, nviews=2, stem_stride1=True).load_state_dict(sd, strict=True)
    note("i_hop16_ckpt_strict_load", f"{os.path.basename(CKPT_DIR)} loaded with strict=True "
                                      f"({len(sd)} tensors)")
    # a SwiGLU model must REFUSE that checkpoint rather than silently drop the FFN weights --
    # the erp FFN keys genuinely differ, and a quiet strict=False load would train from a
    # half-initialised block and be unattributable
    for mode in ("wide", "matched"):
        torch.manual_seed(0)
        mm = OAAv2Depth(C=256, nviews=2, stem_stride1=True, ffn_swiglu=mode)
        out = mm.load_state_dict(sd, strict=False)
        miss = sorted(k for k in out.missing_keys if ".m." in k)
        unexp = sorted(k for k in out.unexpected_keys if ".m." in k)
        assert len(miss) == 8 * N_ERP_BLOCKS, (mode, len(miss))     # n/gate/up/down x weight+bias
        assert len(unexp) == 6 * N_ERP_BLOCKS, (mode, len(unexp))   # m.0/m.1/m.3 x weight+bias
        try:
            mm.load_state_dict(sd, strict=True)
            raise RuntimeError(f"swiglu {mode} should have refused the reference checkpoint")
        except RuntimeError as e:
            assert "Missing key" in str(e) or "missing keys" in str(e).lower(), str(e)[:200]
    note("i_swiglu_refuses_reference_ckpt",
         f"both modes: {8 * N_ERP_BLOCKS} missing / {6 * N_ERP_BLOCKS} unexpected erp FFN "
         f"tensors, strict=True raises (cells train from scratch, as intended)")


# ================================================== part (ii): the flag ON is sane and learnable
def test_part_ii_forward_is_sane_in_both_modes():
    """(B,1,256,512) in [0,1], finite, for both modes at N = 2/4/8, and the output must actually
    differ from the reference -- a swapped nonlinearity that changed nothing would mean the block
    was never reached."""
    shifts = {}
    for nv in (2, 4, 8):
        off = _small_model(nv, stem_stride1=True)
        x = _spec(2, nv)
        base = off(x)
        for mode in ("wide", "matched"):
            m = _small_model(nv, stem_stride1=True, ffn_swiglu=mode)
            D = m(x)
            assert D.shape == (2, 1, 256, 512), (nv, mode, D.shape)
            assert torch.isfinite(D).all(), (nv, mode)
            assert float(D.min()) >= 0.0 and float(D.max()) <= 1.0, (nv, mode)
            d = _maxdiff(base, D)
            shifts[(nv, mode)] = round(d, 6)
            assert d > 1e-9, (nv, mode, "swiglu changed nothing; the erp block was not reached")
    note("ii_forward_shift_vs_reference", {f"N{k[0]}/{k[1]}": v for k, v in shifts.items()})


def test_part_ii_every_gradient_is_alive_and_the_zero_set_is_enumerated():
    """The real (ii) gate. Enumerate the parameters whose gradient is exactly zero (or absent) and
    REPORT the set rather than asserting it away; then require that the four new FFN tensors per
    block are not in it. The reference model's own zero/absent set is measured first, so the
    comparison is against what the architecture already does, not against an ideal."""
    rows = {}
    for mode in ("off", "wide", "matched"):
        nv = 2
        m = _small_model(nv, stem_stride1=True, ffn_swiglu=mode).train()
        m.zero_grad()
        D = m(_spec(2, nv))
        (D * torch.linspace(0.3, 1.7, D.numel(), dtype=D.dtype).view_as(D)).sum().backward()
        absent = sorted(n for n, p in m.named_parameters() if p.grad is None)
        zero = sorted(n for n, p in m.named_parameters()
                      if p.grad is not None and float(p.grad.abs().sum()) == 0.0)
        rows[mode] = {"grad_absent": absent, "grad_exactly_zero": zero}
        if mode != "off":
            for b in range(N_ERP_BLOCKS):
                for t in ("n", "gate", "up", "down"):
                    for w in ("weight", "bias"):
                        q = f"erp.{b}.m.{t}.{w}"
                        g = dict(m.named_parameters())[q].grad
                        assert g is not None, (mode, q, "no gradient at all")
                        assert torch.isfinite(g).all(), (mode, q)
                        assert float(g.abs().sum()) > 0.0, (mode, q, "gradient is exactly zero")
    note("ii_grad_absent_or_zero_per_mode", rows)
    # the zero/absent set must not GROW when the switch is turned on
    for mode in ("wide", "matched"):
        for k in ("grad_absent", "grad_exactly_zero"):
            new = set(rows[mode][k]) - set(rows["off"][k])
            assert not new, (mode, k, sorted(new))
    note("ii_no_new_dead_parameters",
         f"reference absent={len(rows['off']['grad_absent'])} "
         f"zero={len(rows['off']['grad_exactly_zero'])}; neither grows in either mode")


def test_part_ii_measured_parameter_counts_at_the_real_width():
    """The numbers to report, measured at C=256 / 2 observations -- NOT the +0.5 M estimate.
    D1 is allowed over the 15.45 M ceiling; D2 must be parameter-neutral and under it."""
    torch.manual_seed(0)
    nb = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, stem_stride1=True).parameters())
    out = {}
    for mode in ("wide", "matched"):
        torch.manual_seed(0)
        n = sum(p.numel() for p in OAAv2Depth(C=256, nviews=2, stem_stride1=True,
                                              ffn_swiglu=mode).parameters())
        h = _swiglu_hidden(256, mode)
        per_block = _ffn_params(256, h) - _ref_ffn_params(256)
        assert n - nb == N_ERP_BLOCKS * per_block, (mode, n - nb, per_block)
        out[mode] = {"hidden": h, "per_block_delta": per_block, "total_delta": n - nb,
                     "params_M": round(n / 1e6, 4)}
    note("ii_param_base_M", f"{nb / 1e6:.4f} (stem1, C=256, 2 obs)")
    note("ii_param_measured", out)
    # D1: hidden untouched, ceiling deliberately broken
    assert out["wide"]["hidden"] == 512, out["wide"]
    assert out["wide"]["params_M"] > CEILING_M, (
        f"D1 was expected to break the {CEILING_M} M ceiling; it did not, so the scope is wrong")
    # D2: parameter-neutral, and never OVER the reference
    assert out["matched"]["per_block_delta"] <= 0, out["matched"]
    assert abs(out["matched"]["total_delta"]) / nb < 1e-4, out["matched"]
    assert out["matched"]["params_M"] <= CEILING_M, out["matched"]
    note("ii_d1_over_ceiling",
         f"{out['wide']['params_M']:.4f} M vs ceiling {CEILING_M} M -- over by "
         f"{out['wide']['params_M'] - CEILING_M:.4f} M, approved by Changwoo 2026-10-16")
    note("ii_d2_parameter_neutrality",
         f"hidden {out['matched']['hidden']}, {out['matched']['total_delta']:+d} parameters "
         f"({100.0 * out['matched']['total_delta'] / nb:+.5f} %) vs the reference")


def test_part_ii_matched_width_formula_is_the_tightest_fit():
    """D2's whole claim is parameter neutrality, so the width must be the largest one that still
    fits -- one more must overshoot. Checked at several C so the formula, not one lucky number,
    is what is being relied on."""
    rows = {}
    for C in (64, 128, 256, 384):
        h = _swiglu_hidden(C, "matched")
        ref = _ref_ffn_params(C)
        assert _ffn_params(C, h) <= ref, (C, h)
        assert _ffn_params(C, h + 1) > ref, (C, h, "not the tightest fit")
        rows[C] = {"hidden": h, "ref_ffn": ref, "swiglu_ffn": _ffn_params(C, h),
                   "slack": ref - _ffn_params(C, h)}
    note("ii_matched_width_is_tightest", rows)
    assert _swiglu_hidden(256, "wide") == 512


def test_part_ii_only_the_erp_blocks_changed():
    """Scope discipline: the 4 query-refinement blocks and nothing else. The intra (CondSelfAttn),
    inter (InterMicAttn) and ray-attention FFNs must keep their reference shapes, or the cell is
    testing a different experiment from the one that was pre-registered."""
    for mode in ("wide", "matched"):
        m = _small_model(8, ffn_swiglu=mode)
        for blk in m.erp:
            assert isinstance(blk.m, _SwiGLUFFN), type(blk.m)
        for blk in m.intra:
            assert isinstance(blk.m, torch.nn.Sequential), ("intra", type(blk.m))
        for blk in m.inter:
            assert isinstance(blk.m, torch.nn.Sequential), ("inter", type(blk.m))
        assert isinstance(m.ray_mic.ffn, torch.nn.Sequential), type(m.ray_mic.ffn)
        changed = sorted({k.split(".m.")[0] for k in m.state_dict()
                          if ".m." in k and any(t in k for t in (".gate.", ".up.", ".down."))})
        assert changed == [f"erp.{i}" for i in range(N_ERP_BLOCKS)], changed
    note("ii_scope_is_the_four_erp_blocks",
         "erp.0..erp.3 swapped; intra / inter / ray_mic FFNs untouched in both modes")


def test_part_ii_bad_mode_is_refused():
    """A typo must raise, not silently fall back to the reference and produce a cell that looks
    like a SwiGLU run and is not."""
    for bad in ("swiglu", "geglu", "wide2", ""):
        try:
            _small_model(2, ffn_swiglu=bad)
            raise RuntimeError(f"ffn_swiglu={bad!r} should have been refused")
        except AssertionError as e:
            assert "ffn_swiglu" in str(e) or "swiglu mode" in str(e), str(e)
    note("ii_bad_mode_refused", "swiglu / geglu / wide2 / '' all raise")


def test_part_ii_configs_round_trip_through_core_ckpt():
    """Save {state_dict, args} -> core.ckpt.build(args, DM) -> strict load -> identical prediction.
    Without this both cells train fine and are then UNSCOREABLE: eval.py catches the strict-load
    failure per run and prints only `[skip <run>]`."""
    from core.ckpt import build
    DM = types.SimpleNamespace(IN_CH={"r2": 2, "cB": 4, "fb": 4, "r6": 6, "r8": 8}, POSES={},
                               WINDOW=0, HOP=160, WIN=400, N_FFT=512)
    DM.__name__ = "data_0422"
    cfgs = [{}, {"ffn_swiglu": "wide"}, {"ffn_swiglu": "matched"},
            {"ffn_swiglu": "matched", "qk_norm": True},
            {"ffn_swiglu": "wide", "geo_value_film": True}]
    for extra in cfgs:
        args = dict(nviews=2, data_mode="r2", dim=64, rounds=2, lift_h=4, lift_w=8,
                    stem_stride1=True, max_depth=10.0, dec_deep=True, rounds_wired=True,
                    cond_mode="adaln", full_res_enc=True, multi_scale_lift=True,
                    data_module="data_0422", **extra)
        torch.manual_seed(0)
        src = OAAv2Depth(C=64, nviews=2, lh=4, lw=8, stem_stride1=True,
                         qk_norm=extra.get("qk_norm", False),
                         geo_value_film=extra.get("geo_value_film", False),
                         ffn_swiglu=extra.get("ffn_swiglu", "off")).eval()
        rebuilt, dmode, nch, kind, poses = build(args, DM)
        assert set(rebuilt.state_dict()) == set(src.state_dict()), extra
        rebuilt.load_state_dict(src.state_dict(), strict=True)
        rebuilt.eval()
        x = torch.rand(1, 2, 24, 40)
        with torch.no_grad():
            d = _maxdiff(src(x), rebuilt(x))
        assert d == 0.0, (extra, d)
    note("ii_swiglu_ckpt_round_trip", f"{len(cfgs)} configs strict-loaded and prediction-compared: "
                                      f"max|diff| = 0.0")


def test_part_ii_silu_is_the_gate_and_the_algebra_is_what_it_claims():
    """The block must compute down(silu(gate(h)) * up(h)) on the LayerNormed input -- checked
    against a hand-written reference, so "SwiGLU" is a measured description and not a label."""
    torch.manual_seed(0)
    f = _SwiGLUFFN(32, 48).double().eval()
    x = torch.randn(3, 7, 32, dtype=torch.float64)
    h = f.n(x)
    want = f.down(torch.nn.functional.silu(f.gate(h)) * f.up(h))
    d = _maxdiff(f(x), want)
    note("ii_swiglu_algebra_maxdiff", f"{d:.3e} (against a hand-written down(silu(gate)*up))")
    assert d == 0.0, d
    assert f.hidden == 48
    # and it is NOT equal to the GELU block it replaces, at any shared init -- the point of (ii)
    torch.manual_seed(0)
    ref = torch.nn.Sequential(torch.nn.LayerNorm(32), torch.nn.Linear(32, 64), torch.nn.GELU(),
                              torch.nn.Linear(64, 32)).double().eval()
    assert _maxdiff(f(x), ref(x)) > 1e-6


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
