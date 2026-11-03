"""--pair-feat (core/pairfeat.py) contract tests. CPU only.
  pytest -q tests/test_pairfeat.py
"""
import os, sys, math, tempfile, contextlib
import torch
try:
    import pytest
except ImportError:                                         # shared_audio env has no pytest: run as a script
    class pytest:                                           # noqa: N801
        @staticmethod
        @contextlib.contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError(f"expected {exc.__name__}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import pairfeat                                   # noqa: E402
from model.oaa import OAAv2Depth                            # noqa: E402

R2 = [(0, 0), (0, 1)]
R8 = [(o, e) for o in (0, 1, 2, 3) for e in (0, 1)]
AL = [(0, 0), (1, 0), (2, 0), (3, 0)]                       # single-ear mode: no partners
CB = [(0, 0), (0, 1), (1, 1), (3, 0)]                       # mixed: one pair + two singles


def _spec(N, B=2, H=8, W=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, N, H, W, generator=g) + 1e-3


def test_none_is_the_same_object():
    x = _spec(2)
    assert pairfeat.build(x, R2, "none") is x
    assert pairfeat.in_ch("none") == 1 and pairfeat.in_ch("earmean") == 1
    assert pairfeat.in_ch("ild") == 2 and pairfeat.in_ch("partner") == 2


def test_partner_index():
    assert pairfeat.partner_index(R2) == [1, 0]
    assert pairfeat.partner_index(R8) == [1, 0, 3, 2, 5, 4, 7, 6]
    assert pairfeat.partner_index(AL) == [-1, -1, -1, -1]
    assert pairfeat.partner_index(CB) == [1, 0, -1, -1]


def test_ild_antisymmetric_bounded_and_monotone_in_ild_db():
    x = _spec(8)
    y = pairfeat.build(x, R8, "ild")
    assert y.shape == (2, 16, 8, 16)
    mag, c = y[:, 0::2], y[:, 1::2]
    assert torch.equal(mag, x)                                           # plane 1 = magnitude, untouched
    for i, j in enumerate(pairfeat.partner_index(R8)):
        assert torch.allclose(c[:, i], -c[:, j])                         # c_L = -c_R
    assert float(c.abs().max()) <= 1.0
    # c == tanh(ILD_dB * ln10 / 40) up to EPS
    ild_db = 20 * torch.log10(x[:, 0] / x[:, 1])
    assert torch.allclose(c[:, 0], torch.tanh(ild_db * math.log(10) / 40), atol=1e-4)


def test_partner_plane_is_the_other_ear_and_singles_are_zero():
    x = _spec(4)
    y = pairfeat.build(x, CB, "partner")
    assert torch.equal(y[:, 1], x[:, 1]) and torch.equal(y[:, 3], x[:, 0])
    assert torch.equal(y[:, 5], torch.zeros_like(x[:, 0])) and torch.equal(y[:, 7], torch.zeros_like(x[:, 0]))
    z = pairfeat.build(x, CB, "ild")
    assert torch.equal(z[:, 5], torch.zeros_like(x[:, 0])) and torch.equal(z[:, 7], torch.zeros_like(x[:, 0]))


def test_earmean_matches_the_diagnostic_perturbation():
    x = _spec(8)
    y = pairfeat.build(x, R8, "earmean")
    assert y.shape == x.shape
    for i, j in enumerate(pairfeat.partner_index(R8)):
        assert torch.allclose(y[:, i], 0.5 * (x[:, i] + x[:, j]))
        assert torch.allclose(y[:, i], y[:, j])
    w = pairfeat.build(x[:, :4], AL, "earmean")
    assert torch.equal(w, x[:, :4])                                      # no partner -> unchanged


def test_dropped_observation_leaves_no_leak():
    x = _spec(8)
    y = pairfeat.build(x, R8, "ild", dropped=[2])                        # drop (slot 1, L); partner = 3
    assert torch.equal(y[:, 4], torch.zeros_like(x[:, 0])) and torch.equal(y[:, 5], torch.zeros_like(x[:, 0]))
    assert torch.equal(y[:, 7], torch.zeros_like(x[:, 0]))               # partner's pair plane zeroed
    assert torch.equal(y[:, 6], x[:, 3])                                 # partner's magnitude kept
    ref = pairfeat.build(x, R8, "ild")
    keep = [k for k in range(16) if k not in (4, 5, 7)]
    assert torch.equal(y[:, keep], ref[:, keep])                         # nothing else touched
    e = pairfeat.build(x, R8, "earmean", dropped=[0])
    assert torch.equal(e[:, 0], torch.zeros_like(x[:, 0]))
    assert torch.allclose(e[:, 1], 0.5 * (x[:, 0] + x[:, 1]))


def test_model_in_ch_2_forward_and_param_delta():
    SMALL = dict(C=64, lh=4, lw=8, enc_res=(16, 32))                     # same tiny config as tests/test_oaa_modules.py
    torch.manual_seed(0)
    m1 = OAAv2Depth(nviews=2, in_ch=1, **SMALL)
    torch.manual_seed(0)
    m2 = OAAv2Depth(nviews=2, in_ch=2, **SMALL)
    n1 = sum(p.numel() for p in m1.parameters()); n2 = sum(p.numel() for p in m2.parameters())
    assert n2 - n1 == (64 // 2) * 1 * 3 * 3                              # stem conv gains one input plane (+288 at ngf 64)
    x = _spec(2, B=1, H=16, W=32)
    y = pairfeat.build(x, R2, "ild")
    m2.eval()
    with torch.no_grad():
        out = m2(y, view_poses=[(0.0, -1.0), (0.0, 1.0)])
    assert out.dim() == 4 and out.shape[:2] == (1, 1)
    assert torch.isfinite(out).all()
    with pytest.raises(AssertionError):
        m2(x, view_poses=[(0.0, -1.0), (0.0, 1.0)])                      # 2 planes expected, 1 given -> loud


def _ipd_planes(N, B=2, H=8, W=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, N, 2, H, W, generator=g) * 2 - 1


def test_ildipd_is_ild_plus_ipd_planes():
    """ildipd (2026-11-02, E135 bin4): [mag, ILD contrast, cos, sgn*sin] = ild's planes + ipd's planes."""
    assert pairfeat.in_ch("ipd") == 3 and pairfeat.in_ch("ildipd") == 4
    assert pairfeat.needs_ipd("ipd") and pairfeat.needs_ipd("ildipd")
    assert not any(pairfeat.needs_ipd(k) for k in pairfeat.KINDS if k not in ("ipd", "ildipd"))
    assert not pairfeat.folds_views("ildipd")
    x, ipd = _spec(8), _ipd_planes(8)
    y = pairfeat.build(x, R8, "ildipd", ipd=ipd)
    assert y.shape == (2, 32, 8, 16)
    ild = pairfeat.build(x, R8, "ild")
    ph = pairfeat.build(x, R8, "ipd", ipd=ipd)
    assert torch.equal(y[:, 0::4], x)                                    # plane 0 = magnitude, untouched
    assert torch.equal(y[:, 1::4], ild[:, 1::2])                         # plane 1 = ild's contrast, bit for bit
    assert torch.equal(y[:, 2::4], ph[:, 1::3]) and torch.equal(y[:, 3::4], ph[:, 2::3])   # planes 2,3 = ipd's
    with pytest.raises(AssertionError):
        pairfeat.build(x, R8, "ildipd")                                  # no loader planes -> loud
    # singles (no partner) get three zero planes, their magnitude survives
    z = pairfeat.build(x[:, :4], CB, "ildipd", ipd=ipd[:, :4])
    assert torch.equal(z[:, 8], x[:, 2]) and float(z[:, 9:12].abs().sum()) == 0.0
    assert torch.equal(z[:, 12], x[:, 3]) and float(z[:, 13:16].abs().sum()) == 0.0


def test_ildipd_dropped_observation_leaves_no_leak():
    x, ipd = _spec(8), _ipd_planes(8)
    y = pairfeat.build(x, R8, "ildipd", dropped=[2], ipd=ipd)            # drop (slot 1, L); partner = 3
    assert float(y[:, 8:12].abs().sum()) == 0.0                          # all four planes of obs 2
    assert float(y[:, 13:16].abs().sum()) == 0.0                         # partner's three pair planes
    assert torch.equal(y[:, 12], x[:, 3])                                # partner's magnitude kept
    ref = pairfeat.build(x, R8, "ildipd", ipd=ipd)
    keep = [k for k in range(32) if k not in (8, 9, 10, 11, 13, 14, 15)]
    assert torch.equal(y[:, keep], ref[:, keep])                         # nothing else touched


def test_ildmin_is_the_unsigned_ild_and_symmetric_within_the_pair():
    """ildmin (2026-11-03): plane 2 = min/max level ratio in [0,1], same value for both ears, in_ch 2."""
    assert pairfeat.in_ch("ildmin") == 2 and not pairfeat.needs_ipd("ildmin") and not pairfeat.folds_views("ildmin")
    x = _spec(8)
    y = pairfeat.build(x, R8, "ildmin")
    assert y.shape == (2, 16, 8, 16)
    mag, c = y[:, 0::2], y[:, 1::2]
    assert torch.equal(mag, x)                                           # plane 1 = magnitude, untouched
    for i, j in enumerate(pairfeat.partner_index(R8)):
        assert torch.equal(c[:, i], c[:, j])                             # both ears receive the SAME plane
    assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0
    # min/max == exp(-|ln(own/partner)|) == (1 - |c_ild|) / (1 + |c_ild|): only the sign of `ild` is gone
    ild = pairfeat.build(x, R8, "ild")[:, 1::2]
    assert torch.allclose(c, (1 - ild.abs()) / (1 + ild.abs()), atol=1e-4)
    assert torch.allclose(c[:, 0], torch.exp(-torch.log(x[:, 0] / x[:, 1]).abs()), atol=1e-4)
    # a channel without a partner gets a zero plane, exactly like ild
    z = pairfeat.build(x[:, :4], CB, "ildmin")
    assert torch.equal(z[:, 5], torch.zeros_like(x[:, 0])) and torch.equal(z[:, 7], torch.zeros_like(x[:, 0]))
    assert torch.equal(z[:, 1], z[:, 3]) and torch.equal(z[:, 0], x[:, 0]) and torch.equal(z[:, 2], x[:, 1])


def test_ildmin_dropped_observation_leaves_no_leak():
    x = _spec(8)
    y = pairfeat.build(x, R8, "ildmin", dropped=[2])                     # drop (slot 1, L); partner = 3
    assert torch.equal(y[:, 4], torch.zeros_like(x[:, 0])) and torch.equal(y[:, 5], torch.zeros_like(x[:, 0]))
    assert torch.equal(y[:, 7], torch.zeros_like(x[:, 0]))               # partner's pair plane zeroed
    assert torch.equal(y[:, 6], x[:, 3])                                 # partner's magnitude kept
    ref = pairfeat.build(x, R8, "ildmin")
    keep = [k for k in range(16) if k not in (4, 5, 7)]
    assert torch.equal(y[:, keep], ref[:, keep])                         # nothing else touched


def test_ildmin_model_in_ch_2_forward():
    SMALL = dict(C=64, lh=4, lw=8, enc_res=(16, 32))
    torch.manual_seed(0)
    m = OAAv2Depth(nviews=2, in_ch=pairfeat.in_ch("ildmin"), **SMALL)
    x = _spec(2, B=1, H=16, W=32)
    y = pairfeat.build(x, R2, "ildmin")
    assert y.shape == (1, 4, 16, 32)
    m.eval()
    with torch.no_grad():
        out = m(y, view_poses=[(0.0, -1.0), (0.0, 1.0)])
    assert out.shape[:2] == (1, 1) and torch.isfinite(out).all()


def test_ckpt_roundtrip_strict_load_via_core_ckpt_build():
    import data_0422 as DM
    from core.ckpt import build
    torch.manual_seed(0)
    m = OAAv2Depth(C=256, nviews=2, in_ch=2)
    args = {"nviews": 2, "data_mode": "r2", "dim": 256, "rounds": 2, "lift_h": 16, "lift_w": 32,
            "rounds_wired": True, "pair_feat": "ild", "stft_hop": DM.HOP, "stft_win": DM.WIN,
            "stft_nfft": DM.N_FFT, "stft_window": DM.WINDOW, "data_module": "data_0422"}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "best.pth")
        torch.save({"state_dict": m.state_dict(), "args": args}, p)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m2, dmode, nch, kind, poses = build(ck["args"], DM)
        m2.load_state_dict(ck["state_dict"], strict=True)
        assert (dmode, nch, kind) == ("r2", 2, "spec") and len(poses) == 2
        # a "none" checkpoint still rebuilds with in_ch=1
        args2 = dict(args, pair_feat="none")
        m3, *_ = build(args2, DM)
        assert m3.enc.stem[0].in_channels == 1 and m2.enc.stem[0].in_channels == 2


if __name__ == "__main__":                                  # plain-python runner (no pytest in shared_audio)
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:                          # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {e!r}")
    print("ALL-PASS" if not fails else f"FAILED={fails}")
    sys.exit(1 if fails else 0)
