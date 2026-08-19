"""Test-set evaluation for OAA / BatVision / ResNet / ViT / EchoScan checkpoints (model rebuilt from the
checkpoint's saved args). EchoDiffusion is evaluated by eval_echodiffusion.py (isolated env).

Metrics: cos-latitude-weighted, PER-IMAGE (batch-invariant) MAE / RMSE / AbsRel / log10 / delta1-3,
plus near(<3m)/mid(3-6m)/far(>6m) band MAE (core/metrics.py).

Run:  DATA_MODULE=data_0422 python eval.py --run-name oaa_r8_fin bat_r8_fin [--ckpt best] [--compare-dir comparison]
"""
import os, json, argparse
import torch

from core.data import get_data_module
from core.ckpt import resolve_run
from core.evaluate import evaluate
from core.metrics import KEYS, BANDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", nargs="+", required=True)
    ap.add_argument("--out-dir", default="out")             # where our own runs are written
    ap.add_argument("--compare-dir", default="comparison")  # baseline runs + summary output (compare.json)
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    a = ap.parse_args()
    DM = get_data_module()
    device = torch.device("cuda")
    res = {}
    for r in a.run_name:
        try:
            res[r] = evaluate(resolve_run(r, [a.out_dir, a.compare_dir]), DM, a.ckpt, device)
        except Exception as e:
            print(f"[skip {r}] {e}", flush=True)
    cols = KEYS + [b[0] for b in BANDS] + ["Params(M)"]
    print(f"\n{'model':24}" + "".join(f"{c:>10}" for c in cols))
    for r, v in res.items():
        print(f"{r:24}" + "".join(f"{v[c]:10.4f}" for c in cols))
    os.makedirs(a.compare_dir, exist_ok=True)
    cj = os.path.join(a.compare_dir, "compare.json")
    merged = {}
    if os.path.exists(cj):
        try: merged = json.load(open(cj))
        except Exception: merged = {}
    merged.update(res)                      # merge: keep existing rows, overwrite re-evaluated ones
    json.dump(merged, open(cj, "w"), indent=2)
    print(f"\n[saved] {cj}", flush=True)


if __name__ == "__main__":
    main()
