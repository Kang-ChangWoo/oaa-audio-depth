"""Test eval for EchoDiffusion runs — isolated env (fp32; same metrics as eval.py).

  CUDA_VISIBLE_DEVICES=? DATA_MODULE=data_0422 R0422_SPLIT=off3 HF_HOME=/root/local1/changwoo/_echodiff_weights \
    /root/local1/changwoo/echodiff_env/bin/python eval_echodiffusion.py --run-name eco_r2 eco_fb eco_r6 eco_r8
"""
import os, json, math, argparse, importlib
import torch
from model.echodiffusion import EchoDiffusionDepth

_DM = importlib.import_module(os.environ.get("DATA_MODULE", "data_0422"))
KEYS = ["MAE", "MAE_plain", "RMSE", "AbsRel", "log10", "delta1", "delta2", "delta3"]
BANDS = [("near<3", 0, 3), ("mid3-6", 3, 6), ("far>6", 6, 10)]


def cos_lat(h, device):
    v = torch.arange(h, device=device, dtype=torch.float32)
    return torch.cos((math.pi / 2) - (v + 0.5) / h * math.pi).clamp(min=1e-3)


@torch.no_grad()
def evaluate(run_dir, ckpt, device):
    ck = torch.load(os.path.join(run_dir, f"{ckpt}.pth"), map_location="cpu", weights_only=False)
    a = ck["args"]; mode = a.get("mode", "r2"); md = a.get("max_depth", 10.0)
    model = EchoDiffusionDepth(in_ch=_DM.IN_CH[mode]).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    ld = _DM.spec_wave_loader("test", 16, False, 5, mode)
    wlat = cos_lat(256, device).view(1, 1, 256, 1)
    acc = {k: 0.0 for k in KEYS}; n = 0
    be = {b[0]: [0.0, 0.0] for b in BANDS}
    for b in ld:
        D = (model(b["spec"].to(device), b["wave"][:, :2].to(device)).float() * md)   # fp32
        gt = b["depth"].to(device) * md; mask = b["mask"].to(device)
        w = wlat * mask; B = D.shape[0]
        pi = lambda num, den: (num.flatten(1).sum(1) / den.flatten(1).sum(1).clamp(min=1e-6))
        acc["MAE"] += float(pi((D - gt).abs() * w, w).mean()) * B
        acc["MAE_plain"] += float(pi((D - gt).abs() * mask, mask).mean()) * B
        acc["RMSE"] += float(pi(((D - gt) ** 2) * w, w).clamp(min=0).sqrt().mean()) * B
        acc["AbsRel"] += float(pi((D - gt).abs() / gt.clamp(min=0.1) * w, w).mean()) * B
        acc["log10"] += float(pi((torch.log10(D.clamp(min=0.1)) - torch.log10(gt.clamp(min=0.1))).abs() * w, w).mean()) * B
        rt = torch.maximum(D.clamp(min=0.1) / gt.clamp(min=0.1), gt.clamp(min=0.1) / D.clamp(min=0.1))
        for k, thr in (("delta1", 1.25), ("delta2", 1.25 ** 2), ("delta3", 1.25 ** 3)):
            acc[k] += float(pi((rt < thr).float() * w, w).mean()) * B
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
    ap.add_argument("--compare-dir", default="comparison")
    ap.add_argument("--ckpt", default="best")
    a = ap.parse_args()
    device = torch.device("cuda")
    res = {}
    for r in a.run_name:
        try:
            res[r] = evaluate(os.path.join(a.compare_dir, r), a.ckpt, device)
        except Exception as e:
            print(f"[skip {r}] {e}", flush=True)
    cols = KEYS + [b[0] for b in BANDS] + ["Params(M)"]
    print(f"\n{'model':16}" + "".join(f"{c:>10}" for c in cols))
    for r, v in res.items():
        print(f"{r:16}" + "".join(f"{v[c]:10.4f}" for c in cols))
    json.dump(res, open(os.path.join(a.compare_dir, "compare_eco.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
