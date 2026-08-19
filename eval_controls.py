"""Eval-time mechanism controls (ablation): audio-shuffle / pose-shuffle / LR-swap.

Perturbs a representative checkpoint without retraining to test whether pose/ear information is actually used.
  * audio_shuffle: roll the spectrograms by yaw-pair block (pose labels fixed) — breaks the audio<->pose correspondence
  * pose_shuffle : roll view_poses by pair (audio fixed) — symmetric check
  * lr_swap     : swap L/R audio within a pair (ear labels fixed) — ear-identity check
r2 has a single pair, so shuffle is undefined (lr_swap only).
Run:  DATA_MODULE=data_0422 R0422_SPLIT=off3 python3 eval_controls.py --run-name oaa_r2_fin ... [--out controls.json]
"""
import os, json, math, argparse
import torch

from eval import build, resolve_run, _DM, loader
from train_oaa import cos_lat


def pair_blocks(nch):
    return [(i, i + 1) for i in range(0, nch, 2)]


def perm_audio_pairs(spec, roll):
    """spec (B, nv, H, W): reorder channels with yaw-pair blocks rolled by `roll`."""
    nb = spec.shape[1] // 2
    order = []
    for b in range(nb):
        src = (b + roll) % nb
        order += [2 * src, 2 * src + 1]
    return spec[:, order]


def swap_lr(spec):
    order = []
    for b in range(spec.shape[1] // 2):
        order += [2 * b + 1, 2 * b]
    return spec[:, order]


def roll_poses(poses, roll):
    """Roll poses [(yaw, ear), ...] by pair."""
    nb = len(poses) // 2
    out = []
    for b in range(nb):
        src = (b + roll) % nb
        out += [poses[2 * src], poses[2 * src + 1]]
    return out


@torch.no_grad()
def run_control(model, ld, poses, nch, device, mode, max_depth):
    """Same per-image cos-lat-weighted metrics as eval.py."""
    wlat = cos_lat(256, device).view(1, 1, 256, 1)
    KEYS = ["MAE", "MAE_plain", "RMSE", "AbsRel", "log10", "delta1", "delta2", "delta3"]
    acc = {k: 0.0 for k in KEYS}; n = 0
    for b in ld:
        x = b["spec"][:, :nch].to(device)
        vp = poses
        if mode == "audio_shuffle":
            x = perm_audio_pairs(x, 1)
        elif mode == "pose_shuffle":
            vp = roll_poses(poses, 1)
        elif mode == "lr_swap":
            x = swap_lr(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            D = (model(x, view_poses=vp) if vp is not None else model(x)).float() * max_depth
        gt = b["depth"].to(device) * max_depth; mask = b["mask"].to(device)
        w = wlat * mask; B = D.shape[0]
        pi = lambda num, den: (num.flatten(1).sum(1) / den.flatten(1).sum(1).clamp(min=1e-6))
        acc["MAE"] += float(pi((D - gt).abs() * w, w).mean()) * B
        acc["MAE_plain"] += float(pi((D - gt).abs() * mask, mask).mean()) * B
        acc["RMSE"] += float(pi(((D - gt) ** 2) * w, w).clamp(min=0).sqrt().mean()) * B
        acc["AbsRel"] += float(pi((D - gt).abs() / gt.clamp(min=0.1) * w, w).mean()) * B
        acc["log10"] += float(pi((torch.log10(D.clamp(min=0.1)) - torch.log10(gt.clamp(min=0.1))).abs() * w, w).mean()) * B
        rt = torch.maximum(D.clamp(min=0.1) / gt.clamp(min=0.1), gt.clamp(min=0.1) / D.clamp(min=0.1))
        acc["delta1"] += float(pi((rt < 1.25).float() * w, w).mean()) * B
        acc["delta2"] += float(pi((rt < 1.25 ** 2).float() * w, w).mean()) * B
        acc["delta3"] += float(pi((rt < 1.25 ** 3).float() * w, w).mean()) * B
        n += B
    return {k: round(acc[k] / n, 4) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", nargs="+", required=True)
    ap.add_argument("--out-dir", default="comparison")
    ap.add_argument("--out", default="comparison/controls.json")
    a = ap.parse_args()
    device = torch.device("cuda")
    _bs = int(os.environ.get("EVAL_BS", "4"))
    results = {}
    for r in a.run_name:
        rd = resolve_run(r, [a.out_dir, "comparison", "comparison/nonselected"])
        ck = torch.load(os.path.join(rd, "best.pth"), map_location="cpu", weights_only=False)
        model, dmode, nch, kind, poses = build(ck["args"])
        model.load_state_dict(ck["state_dict"]); model.to(device).eval()
        md = ck["args"].get("max_depth", 10.0)
        ld = loader("test", _bs, False, 5, dmode)
        modes = ["none", "lr_swap"] + (["audio_shuffle", "pose_shuffle"] if nch > 2 else [])
        results[r] = {}
        for m in modes:
            v = run_control(model, ld, poses, nch, device, m, md)
            results[r][m] = v
            print(f"{r:16} {m:14} " + " ".join(f"{k} {v[k]:.4f}" for k in ("MAE", "RMSE", "AbsRel", "delta1")), flush=True)
    merged = {}
    if os.path.exists(a.out):
        try: merged = json.load(open(a.out))
        except Exception: pass
    merged.update(results)
    json.dump(merged, open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
