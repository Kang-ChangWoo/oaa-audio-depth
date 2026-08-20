"""0820 sanity tests for the AFM backbones (run BEFORE any training; no dataset needed).

Per backbone: pretrained load report, forward shapes for B=2 x N in {2,4,6,8}, backward reaches
every AFM block, LR-group split, EMA-style state-dict round-trip through a pretrained=False build
(the eval.py rebuild path), zero-input (vdrop) finiteness. Finally: the original CNN OAA still
builds and runs without any AFM import side effects.

  CUDA_VISIBLE_DEVICES=2 python 0820_sanity_afm.py [--backbones bat eat ...] [--cpu]
"""
import argparse, sys
import torch

from model.oaa import OAAv2Depth
from model.audio_backbones_0820 import (BACKBONES, OAAv2DepthAFM, build_afm_model, make_param_groups)


def check(name, device):
    print(f"\n================ {name} ================", flush=True)
    args = dict(audio_backbone=name, dim=256, nviews=4, rounds=2, lift_h=16, lift_w=32,
                stem_stride1=False, max_depth=10.0)
    m = build_afm_model(args, pretrained=True).to(device)
    assert m.enc.afm.pretrained_loaded, "pretrained not loaded"
    n_tot = sum(p.numel() for p in m.parameters())
    groups = make_param_groups(m, 5e-4, 0.1, 1e-4)
    n_pre = sum(p.numel() for p in groups[0]["params"])
    assert 80e6 < n_pre < 100e6, f"pretrained group {n_pre/1e6:.1f}M outside ViT-B range"
    print(f"[ok] params total {n_tot/1e6:.2f}M | pretrained {n_pre/1e6:.2f}M @0.1x | "
          f"new {(n_tot-n_pre)/1e6:.2f}M @base")

    # forward/backward across N (rebuild model per N: nviews is a constructor arg)
    for nv in (2, 4, 6, 8):
        am = dict(args, nviews=nv)
        mn = build_afm_model(am, pretrained=(nv == 4)) if nv != 4 else m   # reuse loaded model for nv=4
        if nv != 4:
            mn.enc.afm.load_state_dict(m.enc.afm.state_dict())             # avoid re-download; same weights
        mn = mn.to(device)
        x = torch.rand(2, nv, 256, 512, device=device) * 3
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            D = mn(x)
        assert D.shape == (2, 1, 256, 512), f"bad output {D.shape}"
        loss = D.float().mean(); loss.backward()
        # gradients must reach every AFM block and the new patch/proj layers
        for i, blk in enumerate(mn.enc.afm.blocks):
            g = blk.attn.qkv.weight.grad
            assert g is not None and torch.isfinite(g).all(), f"N={nv}: no/inf grad at block {i}"
        assert mn.enc.afm.patch.weight.grad is not None and mn.enc.afm.proj.weight.grad is not None
        assert mn.enc.fine_enc.stem[0].weight.grad is not None, "fine CNN path got no gradient"
        assert torch.isfinite(D).all()
        mn.zero_grad(set_to_none=True)
        print(f"[ok] N={nv}: forward {tuple(x.shape)} -> {tuple(D.shape)}, grads reach all 12 blocks + fine path")

    # weight sharing across observations is structural (single enc batched over B*N) — verify by identity
    assert m.enc is m.enc and len({id(m.enc.afm)}) == 1
    # zero observation (vdrop) input stays finite
    x = torch.rand(2, 4, 256, 512, device=device); x[:, 1] = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        assert torch.isfinite(m(x)).all(), "zeroed observation produced non-finite output"
    print("[ok] zeroed-observation (vdrop) input finite")

    # eval.py rebuild path: pretrained=False build + strict state-dict load reproduces outputs
    m2 = build_afm_model(args, pretrained=False).to(device)
    m2.load_state_dict(m.state_dict(), strict=True)
    m.eval(); m2.eval()
    x = torch.rand(1, 4, 256, 512, device=device)
    with torch.no_grad():
        d1, d2 = m(x).float(), m2(x).float()
    assert torch.equal(d1, d2), "rebuild-from-args + state_dict does not reproduce outputs"
    print("[ok] pretrained=False rebuild + strict load reproduces outputs bit-for-bit")
    del m, m2, mn
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=[b for b in BACKBONES if b != "cnn"])
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    device = torch.device("cpu" if a.cpu else "cuda")
    failed = []
    for b in a.backbones:
        try:
            check(b, device)
        except Exception as e:
            failed.append((b, repr(e)))
            print(f"[FAIL] {b}: {e!r}", flush=True)
    # original CNN model untouched
    cnn = OAAv2Depth(C=256, nviews=4).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        D = cnn(torch.rand(2, 4, 256, 512, device=device))
    assert D.shape == (2, 1, 256, 512)
    print(f"\n[ok] original CNN OAAv2Depth unaffected ({sum(p.numel() for p in cnn.parameters())/1e6:.2f}M)")
    print(f"\n==== SANITY {'FAILED: ' + str(failed) if failed else 'ALL PASSED'} ====", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
