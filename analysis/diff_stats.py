"""Mic-differential diagnostics for the AFM encoder (Differential SSLAM-OAA, spec section 9).

Measures, per run, how much of the per-observation AFM feature is mic-SPECIFIC rather than
common-mode. For tokens S_i of observation i (identical token indexing across mics -- the AFM
never sees pose, so token n is the same time-frequency patch for every mic):

    Sbar = mean_j S_j                      common room / late-reverberation component
    D_i  = S_i - Sbar                      mic-specific component
    dratio  = mean_i ||D_i||_F / ||S_i||_F        how much mic-specific signal survives
    cos     = mean_{i<j} cos(S_i, S_j)            pairwise mic similarity (1.0 = total collapse)
    cos_D   = mean_{i<j} cos(D_i, D_j)            similarity of the differentials themselves

and, for models trained with --mic-diff, the learned gate g_i (channel-wise) mean/std.

The hypothesis under test is that the MP3D 8ch collapse of vanilla sslam is a common-mode
collapse: cos -> 1, dratio -> 0, i.e. the eight observations stop being distinguishable to the
backbone. Nothing here touches the eval pipeline; it is a read-only forward pass.

  DATA_MODULE=data_mp3d MP3D_ROOT=... EVAL_BS=8 CUDA_VISIBLE_DEVICES=3 \
    python analysis/diff_stats.py --run-name 0820_sslamllrd_cs_r8_mp3d --out comparison_0820/diff_stats.json

`--drop-sweep` additionally re-runs one 8-mic model with k channels zeroed (the micdrop protocol),
so the 8/6/4/2 live-mic trend can be read off a single checkpoint instead of four different cells.
"""
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


def _stats(S, gate):
    """S (B,nv,M,C) -> dict of scalars for this batch (sums, weighted by B later)."""
    B, nv = S.shape[:2]
    F = S.reshape(B, nv, -1).float()
    D = (S - S.mean(1, keepdim=True)).reshape(B, nv, -1).float()
    sn, dn = F.norm(dim=-1), D.norm(dim=-1)
    out = {"dratio": float((dn / sn.clamp(min=1e-6)).mean()),
           "snorm": float(sn.mean()), "dnorm": float(dn.mean())}
    for tag, X in (("cos", F), ("cos_D", D)):
        U = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        G = U @ U.transpose(1, 2)                                    # (B,nv,nv)
        iu = torch.triu_indices(nv, nv, offset=1)
        out[tag] = float(G[:, iu[0], iu[1]].mean()) if nv > 1 else float("nan")
    if gate is not None:
        out["gate_mean"] = float(gate.float().mean())
        out["gate_std"] = float(gate.float().std())
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", nargs="+", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--out", default="comparison_0820/diff_stats.json")
    ap.add_argument("--dirs", nargs="+", default=["out", "comparison", "comparison_0820"])
    ap.add_argument("--drop-sweep", action="store_true",
                    help="also report stats with k channels zeroed (8mic model -> 8/6/4/2 live mics)")
    ap.add_argument("--max-batches", type=int, default=0, help="0 = whole test split")
    a = ap.parse_args()
    device = torch.device("cuda")
    saved = json.load(open(a.out)) if os.path.exists(a.out) else {}

    for run in a.run_name:
        rd = resolve_run(run, a.dirs)
        ck = torch.load(os.path.join(rd, f"{a.ckpt}.pth"), map_location="cpu", weights_only=False)
        model, dmode, nch, kind, poses = build(ck["args"], _DM)
        model.load_state_dict(ck["state_dict"]); model.to(device).eval()
        assert kind == "spec", f"{run} is not a spectrogram model"

        cap = {}
        model.enc.register_forward_hook(lambda m, i, o: cap.__setitem__("t", o[0]))
        drops = [("k0", None)]
        if a.drop_sweep and nch > 2:
            rng = random.Random(0)
            drops += [(f"k{k}", tuple(sorted(rng.sample(range(nch), k)))) for k in (2, 4, 6) if k < nch]

        ld = _DM.loader("test", int(os.environ.get("EVAL_BS", "8")), False, 5, dmode)
        acc = {t: {} for t, _ in drops}; n = {t: 0 for t, _ in drops}
        for bi, b in enumerate(ld):
            if a.max_batches and bi >= a.max_batches: break
            x0 = b["spec"][:, :nch].to(device); B = x0.shape[0]
            for tag, idx in drops:
                x = x0
                if idx:
                    x = x0.clone(); x[:, list(idx)] = 0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model(x, view_poses=poses) if poses is not None else model(x)
                S = cap["t"].reshape(B, nch, -1, model.C)
                for k, v in _stats(S, getattr(model, "last_gate", None)).items():
                    acc[tag][k] = acc[tag].get(k, 0.0) + v * B
                n[tag] += B
        rec = {"nviews": nch, "data_mode": dmode, "mic_diff": ck["args"].get("mic_diff", "none"),
               "afm_stem": ck["args"].get("afm_stem", "linear"), "dataset": _DM.__name__,
               "stats": {t: {k: v / n[t] for k, v in acc[t].items()} for t, _ in drops},
               "live_mics": {t: nch - (len(i) if i else 0) for t, i in drops}}
        saved[run] = rec
        json.dump(saved, open(a.out, "w"), indent=2)
        print(f"== {run}  ({_DM.__name__}, {nch} mics, stem={rec['afm_stem']}, mic_diff={rec['mic_diff']})")
        for t, _ in drops:
            s = rec["stats"][t]
            g = f"  gate {s['gate_mean']:.3f}+-{s['gate_std']:.3f}" if "gate_mean" in s else ""
            print(f"  {rec['live_mics'][t]} live mics: dratio {s['dratio']:.4f}  "
                  f"cos {s['cos']:.4f}  cos_D {s['cos_D']:.4f}{g}", flush=True)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
