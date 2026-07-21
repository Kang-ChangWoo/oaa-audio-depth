"""Test evaluation for OAA / BatVision checkpoints (auto-detected from saved args).

Metrics follow the research repo's MetricBank: cos-latitude-weighted, PER-IMAGE (batch-
invariant) MAE / RMSE / AbsRel / delta1, plus near(<3m)/mid(3-6m)/far(>6m) band MAE.

Run:  python3 eval.py --run-name r8_s0 bat_r8_s0 [--ckpt best]
"""
import os, json, math, argparse
import torch

from data import loader, IN_CH
from oaa import OAAv2Depth
from batvision import RotDepth
from train_oaa import cos_lat

KEYS = ["MAE", "RMSE", "AbsRel", "delta1"]
BANDS = [("near<3", 0, 3), ("mid3-6", 3, 6), ("far>6", 6, 10)]


def build(args):
    """Reconstruct the model from a checkpoint's saved args (works for main/ and research-repo ckpts)."""
    if "mode" in args or "feat_c" in args:                                  # batvision
        mode = args.get("mode", "cB")
        m = RotDepth(in_ch=IN_CH[mode], feat_c=args.get("feat_c", 32), ngf=args.get("ngf", 64))
        return m, mode, IN_CH[mode]
    nv = args.get("nviews", 4)
    m = OAAv2Depth(C=args.get("dim", 256), nviews=nv, cond_mode=args.get("cond_mode", "adaln"),
                   max_depth=args.get("max_depth", 10.0))
    return m, {2: "r2", 4: "cB", 6: "r6", 8: "r8"}[nv], nv


@torch.no_grad()
def evaluate(run, out_dir, ckpt, device, max_depth=10.0):
    ck = torch.load(os.path.join(out_dir, run, f"{ckpt}.pth"), map_location="cpu", weights_only=False)
    model, dmode, nch = build(ck["args"])
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    ld = loader("test", 32, False, 5, dmode)
    wlat = cos_lat(256, device).view(1, 1, 256, 1)
    acc = {k: 0.0 for k in KEYS}; n = 0
    be = {b[0]: [0.0, 0.0] for b in BANDS}
    for b in ld:
        sp = b["spec"][:, :nch].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            D = model(sp).float() * max_depth
        gt = b["depth"].to(device) * max_depth; mask = b["mask"].to(device)
        w = wlat * mask; B = D.shape[0]
        pi = lambda num, den: (num.flatten(1).sum(1) / den.flatten(1).sum(1).clamp(min=1e-6))
        acc["MAE"] += float(pi((D - gt).abs() * w, w).mean()) * B
        acc["RMSE"] += float(pi(((D - gt) ** 2) * w, w).clamp(min=0).sqrt().mean()) * B
        acc["AbsRel"] += float(pi((D - gt).abs() / gt.clamp(min=0.1) * w, w).mean()) * B
        rt = torch.maximum(D.clamp(min=0.1) / gt.clamp(min=0.1), gt.clamp(min=0.1) / D.clamp(min=0.1))
        acc["delta1"] += float(pi((rt < 1.25).float() * w, w).mean()) * B
        n += B
        err = (D - gt).abs()
        for nm, lo, hi in BANDS:
            bm = mask * (gt >= lo) * (gt < hi)
            be[nm][0] += (err * bm).sum().item(); be[nm][1] += bm.sum().item()
    out = {k: acc[k] / n for k in KEYS}
    for nm in be:
        out[nm] = be[nm][0] / max(be[nm][1], 1e-6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", nargs="+", required=True)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    a = ap.parse_args()
    device = torch.device("cuda")
    res = {}
    for r in a.run_name:
        try:
            res[r] = evaluate(r, a.out_dir, a.ckpt, device)
        except Exception as e:
            print(f"[skip {r}] {e}", flush=True)
    cols = KEYS + [b[0] for b in BANDS]
    print(f"\n{'model':24}" + "".join(f"{c:>9}" for c in cols))
    for r, v in res.items():
        print(f"{r:24}" + "".join(f"{v[c]:9.4f}" for c in cols))
    json.dump(res, open(os.path.join(a.out_dir, "compare.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
