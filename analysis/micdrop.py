"""Progressive mic-drop curve at inference: zero k of 8 channels (k=0..7), poses truthful.

kany trained with vdrop k in {1..6}, so states down to 2 live mics were seen in training;
k=7 (single mic) is extrapolation. For each k we average 3 fixed-seed random subsets
(exhaustive C(8,k) is too many). One data pass evaluates every variant (loader dominates).

  DATA_MODULE=data_0422 R0422_SPLIT=off3 EVAL_BS=6 CUDA_VISIBLE_DEVICES=7 \
    python analysis/micdrop.py --run-name oaa_r8_kany
"""
# --- repo-root bootstrap: importable root modules (eval, data_*, model) + relative comparison/ paths
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
_os.chdir(ROOT)
import os, json, argparse, random
import torch

from core.data import get_data_module
from core.ckpt import build, resolve_run
_DM = get_data_module()
from core.metrics import cos_lat

DRAWS = 3


def variants_for(nch):
    out = [("k0", None)]
    rng = random.Random(0)
    for k in range(1, nch):
        for d in range(DRAWS):
            out.append((f"k{k}_d{d}", tuple(sorted(rng.sample(range(nch), k)))))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", nargs="+", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--out", default="comparison/eval_micdrop.json")
    a = ap.parse_args()
    device = torch.device("cuda")
    saved = {}
    if os.path.exists(a.out):
        try: saved = json.load(open(a.out))
        except Exception: saved = {}
    for run in a.run_name:
        rd = resolve_run(run, ["out", "comparison"])
        ck = torch.load(os.path.join(rd, f"{a.ckpt}.pth"), map_location="cpu", weights_only=False)
        model, dmode, nch, kind, poses = build(ck["args"], _DM)
        model.load_state_dict(ck["state_dict"]); model.to(device).eval()
        max_depth = ck["args"].get("max_depth", 10.0)
        vs = variants_for(nch)
        ld = _DM.loader("test", int(os.environ.get("EVAL_BS", "6")), False, 5, dmode)
        wlat = cos_lat(256, device).view(1, 1, 256, 1)
        MK = ["MAE", "RMSE", "AbsRel", "log10", "delta1", "delta2", "delta3"]
        acc = {t: {k: 0.0 for k in MK} | {"n": 0} for t, _ in vs}
        for b in ld:
            x0 = b["spec"][:, :nch].to(device)
            gt = b["depth"].to(device) * max_depth; mask = b["mask"].to(device)
            w = wlat * mask; B = x0.shape[0]
            pi = lambda num, den: (num.flatten(1).sum(1) / den.flatten(1).sum(1).clamp(min=1e-6))
            for tag, idx in vs:
                x = x0
                if idx:
                    x = x0.clone(); x[:, list(idx)] = 0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    D = (model(x, view_poses=poses) if poses is not None else model(x)).float() * max_depth
                am = acc[tag]
                am["MAE"] += float(pi((D - gt).abs() * w, w).mean()) * B
                am["RMSE"] += float(pi(((D - gt) ** 2) * w, w).clamp(min=0).sqrt().mean()) * B
                am["AbsRel"] += float(pi((D - gt).abs() / gt.clamp(min=0.1) * w, w).mean()) * B
                am["log10"] += float(pi((torch.log10(D.clamp(min=0.1)) - torch.log10(gt.clamp(min=0.1))).abs() * w, w).mean()) * B
                rt = torch.maximum(D.clamp(min=0.1) / gt.clamp(min=0.1), gt.clamp(min=0.1) / D.clamp(min=0.1))
                for i, dk in enumerate(["delta1", "delta2", "delta3"], 1):
                    am[dk] += float(pi((rt < 1.25 ** i).float() * w, w).mean()) * B
                am["n"] += B
        res = {t: acc[t]["MAE"] / acc[t]["n"] for t, _ in vs}
        res_full = {t: {k: acc[t][k] / acc[t]["n"] for k in MK} for t, _ in vs}
        curve = {"k0": res["k0"]}
        for k in range(1, nch):
            ds = [res[f"k{k}_d{d}"] for d in range(DRAWS)]
            curve[f"k{k}"] = {"mean": sum(ds) / len(ds), "min": min(ds), "max": max(ds),
                              "draws": {f"d{d}": ds[d] for d in range(DRAWS)}}
        curve_full = {"k0": res_full["k0"]}
        for k in range(1, nch):
            curve_full[f"k{k}"] = {m: sum(res_full[f"k{k}_d{d}"][m] for d in range(DRAWS)) / DRAWS for m in MK}
        saved[run] = {"curve": curve, "curve_full": curve_full, "subsets": {t: list(i) for t, i in vs if i}}
        json.dump(saved, open(a.out, "w"), indent=2)
        print(f"== {run} (MAE, remaining mics = {nch}-k)")
        print(f"  k0(8mic): {res['k0']:.4f}")
        for k in range(1, nch):
            c = curve[f"k{k}"]
            print(f"  k{k}({nch-k}mic): mean {c['mean']:.4f}  [{c['min']:.4f}~{c['max']:.4f}]", flush=True)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
