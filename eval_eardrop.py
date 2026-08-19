"""Eval-time ear REMOVAL: zero out one ear's channels entirely (poses kept truthful).

r8 channel layout = [0L,0R,90L,90R,180L,180R,270L,270R]  ->  L ears = 0,2,4,6 / R = 1,3,5,7.
Unlike ear-blind (pose sign lied), this physically drops half the microphones — the state
vdrop-trained models saw during training, so the comparison is fair at inference.

  DATA_MODULE=data_0422 R0422_SPLIT=off3 EVAL_BS=6 CUDA_VISIBLE_DEVICES=7 \
    python3 eval_eardrop.py --run-name oaa_r8_kany oaa_r8_vw_rd3
"""
import os, json, argparse
import torch

import eval as ev
from train_oaa import cos_lat

VARIANTS = {"normal": None, "L-only": [1, 3, 5, 7], "R-only": [0, 2, 4, 6]}   # indices zeroed


@torch.no_grad()
def run_eval(run_dir, ckpt, device, zero_idx):
    ck = torch.load(os.path.join(run_dir, f"{ckpt}.pth"), map_location="cpu", weights_only=False)
    model, dmode, nch, kind, poses = ev.build(ck["args"])
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    max_depth = ck["args"].get("max_depth", 10.0)
    ld = ev.loader("test", int(os.environ.get("EVAL_BS", "8")), False, 5, dmode)
    wlat = cos_lat(256, device).view(1, 1, 256, 1)
    acc = {k: 0.0 for k in ev.KEYS}; n = 0
    be = {b[0]: [0.0, 0.0] for b in ev.BANDS}
    for b in ld:
        x = b["spec"][:, :nch].to(device)
        if zero_idx is not None:
            x = x.clone(); x[:, zero_idx] = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            D = (model(x, view_poses=poses) if poses is not None else model(x)).float() * max_depth
        gt = b["depth"].to(device) * max_depth; mask = b["mask"].to(device)
        w = wlat * mask; B = D.shape[0]
        pi = lambda num, den: (num.flatten(1).sum(1) / den.flatten(1).sum(1).clamp(min=1e-6))
        acc["MAE"] += float(pi((D - gt).abs() * w, w).mean()) * B
        acc["MAE_plain"] += float(pi((D - gt).abs() * mask, mask).mean()) * B
        acc["RMSE"] += float(pi(((D - gt) ** 2) * w, w).clamp(min=0).sqrt().mean()) * B
        acc["AbsRel"] += float(pi((D - gt).abs() / gt.clamp(min=0.1) * w, w).mean()) * B
        acc["log10"] += float(pi((torch.log10(D.clamp(min=0.1)) - torch.log10(gt.clamp(min=0.1))).abs() * w, w).mean()) * B
        rt = torch.maximum(D.clamp(min=0.1) / gt.clamp(min=0.1), gt.clamp(min=0.1) / D.clamp(min=0.1))
        for i, dk in enumerate(["delta1", "delta2", "delta3"], 1):
            acc[dk] += float(pi((rt < 1.25 ** i).float() * w, w).mean()) * B
        n += B
        err = (D - gt).abs()
        for nm, lo, hi in ev.BANDS:
            bm = mask * (gt >= lo) * (gt < hi)
            be[nm][0] += (err * bm).sum().item(); be[nm][1] += bm.sum().item()
    out = {k: acc[k] / n for k in ev.KEYS}
    for nm in be:
        out[nm] = be[nm][0] / max(be[nm][1], 1e-6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", nargs="+", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--out", default="comparison/eval_eardrop.json")
    a = ap.parse_args()
    device = torch.device("cuda")
    saved = {}
    if os.path.exists(a.out):
        try: saved = json.load(open(a.out))
        except Exception: saved = {}
    show = ["MAE", "RMSE", "AbsRel", "delta1", "near<3", "mid3-6", "far>6"]
    print(f"{'run':18}{'variant':9}" + "".join(f"{k:>9}" for k in show))
    for run in a.run_name:
        rd = ev.resolve_run(run, ["out", "comparison"])
        for tag, idx in VARIANTS.items():
            out = run_eval(rd, a.ckpt, device, idx)
            saved.setdefault(run, {})[tag] = out
            json.dump(saved, open(a.out, "w"), indent=2)
            print(f"{run:18}{tag:9}" + "".join(f"{out[k]:9.4f}" for k in show), flush=True)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
