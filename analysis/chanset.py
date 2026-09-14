"""Evaluate a trained model with an EXPLICIT set of channels zeroed (poses left truthful).

Unlike analysis/micdrop.py (random subsets by size), this names the channels, so a structural
hypothesis about a particular microphone pair can be tested directly. r8 channel order is
[0L, 0R, 90L, 90R, 180L, 180R, 270L, 270R]; zeroing 4,5 leaves exactly the r6 microphone set.

  DATA_MODULE=data_mp3d python analysis/chanset.py --run-name 0820_sslamllrd_cs_r8_mp3d \
      --drop none 4,5 2,3 6,7 0,1
"""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path: _sys.path.insert(0, ROOT)
_os.chdir(ROOT)
import os, json, argparse
import torch
from core.data import get_data_module
from core.ckpt import build, resolve_run
from core.metrics import cos_lat
_DM = get_data_module()

NAMES = ["0L", "0R", "90L", "90R", "180L", "180R", "270L", "270R"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", nargs="+", required=True)
    ap.add_argument("--drop", nargs="+", default=["none"], help="comma-separated channel indices, or 'none'")
    ap.add_argument("--dirs", nargs="+", default=["out", "comparison", "comparison_0820"])
    ap.add_argument("--out", default="comparison_0820/chanset.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    saved = json.load(open(a.out)) if os.path.exists(a.out) else {}
    sets = [(d, () if d == "none" else tuple(int(x) for x in d.split(","))) for d in a.drop]

    for run in a.run_name:
        rd = resolve_run(run, a.dirs)
        ck = torch.load(os.path.join(rd, "best.pth"), map_location="cpu", weights_only=False)
        model, dmode, nch, kind, poses = build(ck["args"], _DM)
        model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
        md = ck["args"].get("max_depth", 10.0)
        ld = _DM.loader("test", int(os.environ.get("EVAL_BS", "8")), False, 5, dmode)
        w0 = cos_lat(256, dev).view(1, 1, 256, 1)
        acc = {t: [0.0, 0.0, 0] for t, _ in sets}
        for b in ld:
            x0 = b["spec"][:, :nch].to(dev); gt = b["depth"].to(dev) * md; msk = b["mask"].to(dev)
            w = w0 * msk; B = x0.shape[0]
            pi = lambda n, d: (n.flatten(1).sum(1) / d.flatten(1).sum(1).clamp(min=1e-6))
            for tag, idx in sets:
                x = x0
                if idx:
                    x = x0.clone(); x[:, list(idx)] = 0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    D = (model(x, view_poses=poses) if poses is not None else model(x)).float() * md
                acc[tag][0] += float(pi((D - gt).abs() * w, w).mean()) * B
                acc[tag][1] += float(pi(((D - gt) ** 2) * w, w).clamp(min=0).sqrt().mean()) * B
                acc[tag][2] += B
        res = {t: {"MAE": acc[t][0] / acc[t][2], "RMSE": acc[t][1] / acc[t][2],
                   "zeroed": [NAMES[i] for i in idx] if idx else []} for t, idx in sets}
        saved[run] = res
        json.dump(saved, open(a.out, "w"), indent=2)
        base = res["none"]["MAE"] if "none" in res else float("nan")
        print(f"== {run}  ({_DM.__name__}, {nch} mics)")
        for t, _ in sets:
            r = res[t]
            lbl = "all live" if not r["zeroed"] else "zero " + "+".join(r["zeroed"])
            print(f"  {lbl:22s} MAE {r['MAE']:.4f}  ({r['MAE']-base:+.4f})   RMSE {r['RMSE']:.4f}", flush=True)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
