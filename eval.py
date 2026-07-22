"""Test evaluation for OAA / BatVision checkpoints (auto-detected from saved args).

Metrics follow the research repo's MetricBank: cos-latitude-weighted, PER-IMAGE (batch-
invariant) MAE / RMSE / AbsRel / delta1, plus near(<3m)/mid(3-6m)/far(>6m) band MAE.

Run:  python3 eval.py --run-name r8_s0 bat_r8_s0 [--ckpt best]
"""
import os, json, math, argparse
import torch

import importlib
# data module selectable at runtime: DATA_MODULE=data (MP3D, default) | data_0422 (Replica)
_DM = importlib.import_module(os.environ.get("DATA_MODULE", "data"))
loader, wave_loader, IN_CH = _DM.loader, _DM.wave_loader, _DM.IN_CH
from model.oaa import OAAv2Depth
from model.batvision import RotDepth
from model import PretrainedResNet, PretrainedViT, BeyondI2DDepth, EchoScanDepth
from train_oaa import cos_lat

KEYS = ["MAE", "RMSE", "AbsRel", "delta1"]
BANDS = [("near<3", 0, 3), ("mid3-6", 3, 6), ("far>6", 6, 10)]

# comparison baselines from train_baseline.py (ckpt args carry a "model" field)
_BASE_SPEC = {"resnet": PretrainedResNet, "vit": PretrainedViT, "beyond": BeyondI2DDepth}


def build(args):
    """Reconstruct the model from a checkpoint's saved args.

    Returns (model, dmode, nch, kind) where kind is "spec" or "wave". dmode is the loader mode
    for spec models (None for wave models), nch the channels to slice from the spec.
    """
    name = args.get("model")                                                # train_baseline.py ckpts
    if name in _BASE_SPEC:
        in_ch = args.get("in_ch", IN_CH[args.get("mode", "cB")])
        mode = {2: "r2", 4: "cB", 6: "r6", 8: "r8"}[in_ch]
        m = _BASE_SPEC[name](in_ch=in_ch, pretrained=False) if name in ("resnet", "vit") \
            else BeyondI2DDepth(in_ch=in_ch, pretrained_material=False)
        return m, mode, in_ch, "spec"
    if name == "echoscan":
        return EchoScanDepth(in_ch=2, fs=args.get("fs", 48000)), None, 2, "wave"
    if "feat_c" in args or ("mode" in args and "model" not in args):         # batvision
        mode = args.get("mode", "cB")
        m = RotDepth(in_ch=IN_CH[mode], feat_c=args.get("feat_c", 32), ngf=args.get("ngf", 64))
        return m, mode, IN_CH[mode], "spec"
    nv = args.get("nviews", 4)                                              # oaa
    if args.get("full_res"):                                                # ours: fullres decoder upgrade
        from model.oaa_fullres import OAAv2Depth as OAAFullRes
        m = OAAFullRes(C=args.get("dim", 256), nviews=nv, in_ch=1, cond_mode=args.get("cond_mode", "adaln"),
                       enc_res=((256, 512) if args.get("full_res_enc") else (128, 256)),
                       dec_deep=args.get("dec_deep", False), multi_scale_lift=args.get("multi_scale_lift", False),
                       max_depth=args.get("max_depth", 10.0))
    else:
        m = OAAv2Depth(C=args.get("dim", 256), nviews=nv, cond_mode=args.get("cond_mode", "adaln"),
                       max_depth=args.get("max_depth", 10.0))
    return m, {2: "r2", 4: "cB", 6: "r6", 8: "r8"}[nv], nv, "spec"


def resolve_run(run, search_dirs):
    """Find <dir>/<run>/ across search_dirs (ours in out/, baselines in comparison/)."""
    for d in search_dirs:
        if os.path.isdir(os.path.join(d, run)):
            return os.path.join(d, run)
    raise FileNotFoundError(f"run '{run}' not found under {search_dirs}")


@torch.no_grad()
def evaluate(run_dir, ckpt, device, max_depth=10.0):
    ck = torch.load(os.path.join(run_dir, f"{ckpt}.pth"), map_location="cpu", weights_only=False)
    model, dmode, nch, kind = build(ck["args"])
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    ld = wave_loader("test", 32, False, 5) if kind == "wave" else loader("test", 32, False, 5, dmode)
    wlat = cos_lat(256, device).view(1, 1, 256, 1)
    acc = {k: 0.0 for k in KEYS}; n = 0
    be = {b[0]: [0.0, 0.0] for b in BANDS}
    for b in ld:
        x = b["wave"].to(device) if kind == "wave" else b["spec"][:, :nch].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            D = model(x).float() * max_depth
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
    out["Params(M)"] = params_m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", nargs="+", required=True)
    ap.add_argument("--out-dir", default="out")            # "ours" runs (OAA/BatVision)
    ap.add_argument("--compare-dir", default="comparison")  # baseline runs + summary output
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    a = ap.parse_args()
    device = torch.device("cuda")
    search = [a.out_dir, a.compare_dir]
    res = {}
    for r in a.run_name:
        try:
            res[r] = evaluate(resolve_run(r, search), a.ckpt, device)
        except Exception as e:
            print(f"[skip {r}] {e}", flush=True)
    cols = KEYS + [b[0] for b in BANDS] + ["Params(M)"]
    print(f"\n{'model':24}" + "".join(f"{c:>10}" for c in cols))
    for r, v in res.items():
        print(f"{r:24}" + "".join(f"{v[c]:10.4f}" for c in cols))
    os.makedirs(a.compare_dir, exist_ok=True)
    json.dump(res, open(os.path.join(a.compare_dir, "compare.json"), "w"), indent=2)
    print(f"\n[saved] {os.path.join(a.compare_dir, 'compare.json')}", flush=True)


if __name__ == "__main__":
    main()
