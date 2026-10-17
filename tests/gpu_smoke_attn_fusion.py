"""GPU-only smoke for the OAA attention/fusion variants: bf16 autocast, one optimizer step,
production parameter counts, peak VRAM and step/inference timing.

The equivalence proofs live in `tests/test_oaa_attention_variants.py` and run on CPU in float64 on
purpose. This file measures only what a GPU can tell you that a CPU cannot, and it never claims
accuracy. It loads model/oaa.py by path so it does not drag in the `model` package's torchvision
baselines, which lets it run on any box that has torch.

    python tests/gpu_smoke_attn_fusion.py [--nviews 8] [--batch 3] [--json out.json]
"""
import argparse
import importlib.util
import json
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OAA_PATH = os.environ.get("OAA_MODULE", os.path.join(os.path.dirname(HERE), "model", "oaa.py"))
_spec = importlib.util.spec_from_file_location("oaa_standalone", OAA_PATH)
oaa = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(oaa)

VARIANTS = [
    ("E0 reference", {}),
    ("E1 qk-norm", dict(qk_norm=True)),
    ("E2 ray-gate", dict(ray_gate=True)),
    ("E3 qk-norm + ray-gate", dict(qk_norm=True, ray_gate=True)),
    ("E4 hier (equivalence control)", dict(view_fusion="hier")),
    ("E4 hier-geo", dict(view_fusion="hier-geo")),
]


def bench(name, kw, nv, B, device, obs_mask, iters=6):
    torch.manual_seed(0)
    m = oaa.OAAv2Depth(C=256, nviews=nv, **kw).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    spec = torch.rand(B, nv, 257, 251, device=device)
    gt = torch.rand(B, 1, 256, 512, device=device)
    mask = torch.ones_like(gt)
    ov = None
    if obs_mask:
        ov = torch.ones(B, nv, dtype=torch.bool, device=device)
        ov[:, : max(1, nv // 4)] = False
    torch.cuda.reset_peak_memory_stats(device)
    losses, step_ms = [], []
    for i in range(iters):
        t0 = time.perf_counter()
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            D = m(spec, obs_valid=ov)
        loss = ((D.float() - gt).abs() * mask).sum() / mask.sum().clamp(min=1e-6)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
        opt.step()
        torch.cuda.synchronize(device)
        if i >= 2:                                  # drop warmup iterations
            step_ms.append((time.perf_counter() - t0) * 1e3)
        losses.append(float(loss))
        assert torch.isfinite(torch.tensor(losses[-1])), f"{name}: non-finite loss at iter {i}"
        assert torch.isfinite(torch.tensor(gn)), f"{name}: non-finite grad norm at iter {i}"
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
    m.eval()
    with torch.no_grad():
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m(spec, obs_valid=ov)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(5):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(spec, obs_valid=ov)
        torch.cuda.synchronize(device)
        inf_ms = (time.perf_counter() - t0) / 5 * 1e3
    assert torch.isfinite(out).all() and float(out.min()) >= 0 and float(out.max()) <= 1
    res = dict(variant=name, params_m=sum(p.numel() for p in m.parameters()) / 1e6,
               peak_vram_gb=round(peak, 3), train_step_ms=round(sum(step_ms) / len(step_ms), 1),
               infer_ms=round(inf_ms, 1), loss_first=round(losses[0], 5), loss_last=round(losses[-1], 5),
               obs_mask=bool(obs_mask))
    del m, opt, spec, gt, mask
    torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nviews", type=int, default=8)
    p.add_argument("--batch", type=int, default=3)
    p.add_argument("--json", default="")
    p.add_argument("--value-moments-only", action="store_true",
                   help="benchmark reference vs value-moments, including masked observations")
    p.add_argument("--stem-stride1", action="store_true")
    a = p.parse_args()
    assert torch.cuda.is_available(), "this file is the GPU half of the test suite"
    dev = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(dev)} torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"nviews={a.nviews} micro-batch={a.batch} (bf16 autocast, one AdamW step per iteration)\n")
    rows, base = [], None
    variants = ([("reference", {}), ("value-moments", dict(value_moments=True))]
                if a.value_moments_only else VARIANTS)
    for name, kw in variants:
        for obs_mask in ((False, True) if not kw or a.value_moments_only else (False,)):
            label = name + " + obs-mask" if obs_mask else name
            r = bench(label, dict(kw, stem_stride1=a.stem_stride1), a.nviews, a.batch, dev, obs_mask)
            if base is None:
                base = r
            r["dparams_m"] = round(r["params_m"] - base["params_m"], 6)
            r["dvram_gb"] = round(r["peak_vram_gb"] - base["peak_vram_gb"], 3)
            r["dstep_pct"] = round((r["train_step_ms"] / base["train_step_ms"] - 1) * 100, 1)
            rows.append(r)
            print(f"{r['variant']:32s} params={r['params_m']:8.3f}M (+{r['dparams_m']:.6f}) "
                  f"VRAM={r['peak_vram_gb']:6.2f}GB ({r['dvram_gb']:+.2f}) "
                  f"step={r['train_step_ms']:7.1f}ms ({r['dstep_pct']:+.1f}%) infer={r['infer_ms']:6.1f}ms")
    if a.json:
        json.dump({"device": torch.cuda.get_device_name(dev), "torch": torch.__version__,
                   "nviews": a.nviews, "batch": a.batch, "rows": rows},
                  open(a.json, "w"), indent=2)
        print(f"\n-> {a.json}")
    print("\nAll variants: finite loss + finite grad norm over 6 bf16 steps, output in [0,1].")


if __name__ == "__main__":
    main()
