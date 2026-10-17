"""Data-free gates: python tests/test_oaa_value_moments.py.

Set OAA_MOMENT_REF_MODULE to a pristine pre-patch model/oaa.py for the additional
old-code regression gate. No pretrained checkpoint or dataset is downloaded.
"""
import importlib.util
import math
import os
from pathlib import Path
import sys
import types
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.oaa import OAAv2Depth, RayMicAttn, _attention_value_std
from core.ckpt import build

torch.set_num_threads(2)
SMALL = dict(C=32, lh=4, lw=8, enc_res=(16, 32), rounds=1)


def ray(n=2, **kwargs):
    torch.manual_seed(7)
    return RayMicAttn(32, **kwargs).double()


def inputs(n=2):
    g = torch.Generator().manual_seed(11)
    q = torch.randn(2, 7, 32, generator=g, dtype=torch.float64)
    t = torch.randn(2, n * 5, 32, generator=g, dtype=torch.float64)
    d = torch.randn(7, 3, generator=g, dtype=torch.float64)
    d /= d.norm(dim=-1, keepdim=True)
    poses = [(math.pi * (i // 2) / max(1, n // 2), (-1., 1.)[i % 2]) for i in range(n)]
    return q, t, d, poses, 5


class ValueMomentTests(unittest.TestCase):
    def test_parameter_budget_and_rng(self):
        torch.manual_seed(1)
        old = OAAv2Depth()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(1)
        new = OAAv2Depth(value_moments=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in new.parameters()) -
                         sum(p.numel() for p in old.parameters()), 256)
        self.assertEqual(set(new.state_dict()) - set(old.state_dict()), {"ray_mic.moment_scale"})
        for k, v in old.state_dict().items():
            self.assertTrue(torch.equal(v, new.state_dict()[k]), k)

    def test_identity_forward_and_shared_gradients(self):
        for n in (2, 4, 8):
            for masked in (False, True):
                with self.subTest(n=n, masked=masked):
                    old, new = ray(n), ray(n, value_moments=True)
                    x = inputs(n)
                    valid = torch.ones(2, n, dtype=torch.bool) if masked else None
                    if masked:
                        valid[0, 0] = False
                        valid[1, -1] = False
                    a, b = old(*x, obs_valid=valid), new(*x, obs_valid=valid)
                    self.assertTrue(torch.equal(a, b))
                    a.square().sum().backward()
                    b.square().sum().backward()
                    for k, p in old.named_parameters():
                        self.assertTrue(torch.equal(p.grad, dict(new.named_parameters())[k].grad), k)
                    g = new.moment_scale.grad
                    self.assertTrue(torch.isfinite(g).all())
                    self.assertGreater(g.abs().min().item(), 0)

    def test_full_model_identity_n2_n4_n8(self):
        for n in (2, 4, 8):
            with self.subTest(n=n), torch.no_grad():
                torch.manual_seed(2)
                old = OAAv2Depth(nviews=n, **SMALL).double().eval()
                torch.manual_seed(2)
                new = OAAv2Depth(nviews=n, value_moments=True, **SMALL).double().eval()
                x = torch.rand(1, n, 24, 40, dtype=torch.float64)
                a, b = old(x), new(x)
                self.assertEqual(a.shape, (1, 1, 256, 512))
                self.assertTrue(torch.equal(a, b))

    def test_masked_observation_cannot_affect_learned_readout(self):
        model = ray(4, value_moments=True)
        with torch.no_grad():
            model.moment_scale.fill_(0.2)
        q, t, d, poses, m = inputs(4)
        valid = torch.tensor([[False, True, True, True], [True, True, False, False]])
        bad = t.clone().reshape(2, 4, m, 32)
        bad[~valid] = 100 * torch.randn_like(bad[~valid])
        bad = bad.reshape_as(t).requires_grad_(True)
        a = model(q, t, d, poses, m, valid)
        b = model(q, bad, d, poses, m, valid)
        self.assertTrue(torch.equal(a, b))
        b.sum().backward()
        self.assertEqual(bad.grad.reshape(2, 4, m, 32)[~valid].abs().max().item(), 0)

    def test_second_moment_distinguishes_equal_means(self):
        p = torch.full((1, 1, 1, 2), 0.5, dtype=torch.float64)
        v = torch.tensor([[[[-1., 1.], [1., -1.]]]], dtype=torch.float64)
        self.assertTrue(torch.equal(p @ v, p @ (2 * v)))
        a, b = _attention_value_std(p, v), _attention_value_std(p, 2 * v)
        self.assertGreater((b - a).min().item(), 0.99)

    def test_statistic_matches_explicit_weighted_variance(self):
        torch.manual_seed(9)
        p = torch.randn(2, 3, 7, 10, dtype=torch.float64).softmax(-1)
        v = torch.randn(2, 3, 10, 4, dtype=torch.float64)
        mu = p @ v
        variance = (p[..., None] * (v[:, :, None] - mu[..., None, :]).square()).sum(-2)
        want = (variance + 1e-6).sqrt() - 1e-3
        torch.testing.assert_close(_attention_value_std(p, v), want, atol=1e-14, rtol=1e-14)

    def test_constant_values_have_zero_dispersion_and_finite_gradients(self):
        p = torch.randn(2, 3, 7, 10, dtype=torch.float64).softmax(-1)
        v = torch.full((2, 3, 10, 4), 10000., dtype=torch.float64, requires_grad=True)
        out = _attention_value_std(p, v)
        self.assertEqual(out.abs().max().item(), 0)
        out.sum().backward()
        self.assertTrue(torch.isfinite(v.grad).all())

    def test_bfloat16_forward_and_backward(self):
        for masked in (False, True):
            model = ray(4, value_moments=True).float()
            with torch.no_grad():
                model.moment_scale.fill_(0.1)
            q, t, d, poses, m = inputs(4)
            valid = torch.tensor([[False, True, True, True], [True, True, False, False]]) if masked else None
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = model(q.float(), t.float(), d.float(), poses, m, valid)
            self.assertTrue(torch.isfinite(out).all())
            out.square().mean().backward()
            for k, p in model.named_parameters():
                self.assertIsNotNone(p.grad, k)
                self.assertTrue(torch.isfinite(p.grad).all(), k)

    def test_checkpoint_build_and_stft_guard(self):
        dm = types.SimpleNamespace(__name__="fake", IN_CH={"r2": 2}, POSES={},
                                   HOP=16, WIN=400, N_FFT=512, WINDOW=2799)
        args = dict(dim=32, nviews=2, value_moments=True, rounds_wired=True,
                    stft_hop=16, stft_win=400, stft_nfft=512, stft_window=2799, data_module="fake")
        original = build(args, dm)[0]
        with torch.no_grad():
            original.ray_mic.moment_scale.fill_(0.12)
        rebuilt = build(args, dm)[0]
        rebuilt.load_state_dict(original.state_dict(), strict=True)
        self.assertTrue(torch.equal(original.ray_mic.moment_scale, rebuilt.ray_mic.moment_scale))
        with self.assertRaisesRegex(RuntimeError, "STFT_HOP"):
            build(dict(args, stft_hop=160), dm)
        old_args = dict(args)
        del old_args["value_moments"]
        old = build(old_args, dm)[0]
        self.assertFalse(hasattr(old.ray_mic, "moment_scale"))

    def test_incompatible_options_fail(self):
        for kw in (dict(view_fusion="hier"), dict(registers=2), dict(geo_value_film=True),
                   dict(ray_gate=True), dict(film_lse=True)):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                RayMicAttn(32, value_moments=True, **kw)
        with self.assertRaises(ValueError):
            OAAv2Depth(value_moments=True, no_cross=True)

    @unittest.skipUnless(os.environ.get("OAA_MOMENT_REF_MODULE"), "pre-patch reference module not supplied")
    def test_default_off_matches_pristine_module(self):
        spec = importlib.util.spec_from_file_location("moment_pristine", os.environ["OAA_MOMENT_REF_MODULE"])
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        for n in (2, 4, 8):
            for kw in ({}, dict(geo_value_film=True), dict(registers=4), dict(view_fusion="hier")):
                torch.manual_seed(7)
                old = ref.RayMicAttn(32, **kw).double()
                new = ray(n, **kw)
                new.load_state_dict(old.state_dict(), strict=True)
                x = inputs(n)
                a, b = old(*x), new(*x)
                self.assertTrue(torch.equal(a, b))
                a.square().sum().backward()
                b.square().sum().backward()
                for k, p in old.named_parameters():
                    other = dict(new.named_parameters())[k].grad
                    if p.grad is None:
                        self.assertIsNone(other)
                    else:
                        self.assertTrue(torch.equal(p.grad, other), k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
