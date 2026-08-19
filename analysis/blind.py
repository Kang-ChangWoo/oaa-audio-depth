"""Eval-time pose ablation: re-evaluate a trained ckpt with pose info blinded at inference.

No training involved — quantifies how much a *pose-trained* model actually relies on the
pose conditioning (complement to the pose-blind *training* runs).

  DATA_MODULE=data_0422 R0422_SPLIT=off3 EVAL_BS=6 CUDA_VISIBLE_DEVICES=7 \
    python analysis/blind.py --run-name oaa_r8_kany oaa_r8_vw_rd3
"""
# --- repo-root bootstrap: importable root modules (eval, data_*, model) + relative comparison/ paths
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
_os.chdir(ROOT)
import os, json, argparse, tempfile
import torch

from core.data import get_data_module
from core.ckpt import build, resolve_run
from core.evaluate import evaluate
_DM = get_data_module()

KEYS = ["MAE", "RMSE", "AbsRel", "delta1", "near<3", "mid3-6", "far>6"]   # print subset; JSON saves all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", nargs="+", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--out", default="comparison/eval_blind.json")
    a = ap.parse_args()
    device = torch.device("cuda")
    saved = {}
    if os.path.exists(a.out):
        try: saved = json.load(open(a.out))
        except Exception: saved = {}
    print(f"{'run':20}{'variant':12}" + "".join(f"{k:>9}" for k in KEYS))
    for run in a.run_name:
        rd = resolve_run(run, ["out", "comparison"])
        ck = torch.load(os.path.join(rd, f"{a.ckpt}.pth"), map_location="cpu", weights_only=False)
        for tag, patch in [("normal", {}), ("pose-blind", {"pose_blind": True}),
                           ("ear-blind", {"ear_blind": True})]:
            with tempfile.TemporaryDirectory() as td:
                torch.save({**ck, "args": {**ck["args"], **patch}}, os.path.join(td, f"{a.ckpt}.pth"))
                out = evaluate(td, _DM, a.ckpt, device)
            saved.setdefault(run, {})[tag] = out          # full metrics: KEYS + bands + Params(M)
            json.dump(saved, open(a.out, "w"), indent=2)
            print(f"{run:20}{tag:12}" + "".join(f"{out[k]:9.4f}" for k in KEYS), flush=True)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
