"""Cheap screening: freeze a real OAA checkpoint, fit only the moment scales on TRAIN,
and score a fixed scene-balanced VALIDATION subset before/after. Never reads TEST.

This is a restricted adapter diagnostic, not the final architecture experiment. A null
does not rule out a benefit after joint retraining. No validation-time fitting is done.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.ckpt import build, _stft_from_manifest
from core.metrics import MetricAccumulator, cos_lat, depth_loss
from model.oaa import OAAv2Depth

STFT = {"STFT_HOP": "stft_hop", "STFT_WIN": "stft_win",
        "STFT_NFFT": "stft_nfft", "STFT_WINDOW": "stft_window"}


def balanced_indices(samples, count, seed):
    """Round-robin shuffled scene buckets; selection uses no labels or model errors."""
    rng = random.Random(seed)
    buckets = {}
    for i, (scene, _) in enumerate(samples):
        buckets.setdefault(scene, []).append(i)
    for values in buckets.values():
        rng.shuffle(values)
    scenes = sorted(buckets)
    rng.shuffle(scenes)
    chosen = []
    while len(chosen) < min(count, len(samples)):
        for scene in scenes:
            if buckets[scene] and len(chosen) < count:
                chosen.append(buckets[scene].pop())
    return chosen


def amp(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def score(model, loader, device, poses, nv, max_depth, samples):
    model.eval()
    acc = MetricAccumulator(device)
    by_scene = {}
    pooled_num = pooled_den = 0.
    wlat = cos_lat(256, device).view(1, 1, 256, 1)
    for b in loader:
        with amp(device):
            pred = model(b["spec"][:, :nv].to(device), view_poses=poses).float() * max_depth
        gt, mask = b["depth"].to(device) * max_depth, b["mask"].to(device)
        acc.update(pred, gt, mask)
        weight = mask * wlat
        pooled_num += ((pred - gt).abs() * weight).sum().item()
        pooled_den += weight.sum().item()
        for j, index in enumerate(b["idx"].tolist()):
            scene = samples[index][0]
            if scene not in by_scene:
                by_scene[scene] = MetricAccumulator(device)
            by_scene[scene].update(pred[j:j+1], gt[j:j+1], mask[j:j+1])
    return {"metrics": acc.result(), "per_scene": {s: a.result() for s, a in by_scene.items()},
            "selection_mae_pooled_m": pooled_num / max(pooled_den, 1e-6)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-samples", type=int, default=256)
    p.add_argument("--val-samples", type=int, default=256)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if min(a.train_samples, a.val_samples, a.passes, a.batch_size) < 1:
        p.error("sample counts, passes and batch size must be positive")
    if a.out_dir.resolve().is_relative_to(REPO):
        p.error("outputs must be outside the repository")
    if a.out_dir.exists():
        p.error("output directory already exists; refusing to overwrite a probe")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device)
    ck = torch.load(a.baseline_run / "best.pth", map_location="cpu", weights_only=False)
    args = _stft_from_manifest(a.baseline_run, dict(ck["args"]))
    dm_name = args.get("data_module")
    if dm_name not in ("data_0422", "data_mp3d"):
        p.error("checkpoint must record data_module=data_0422 or data_mp3d")
    for env, key in STFT.items():
        if args.get(key) is None:
            p.error(f"missing {key}: provide the baseline's original manifest.json")
        os.environ[env] = str(args[key])
    if args.get("value_moments") or args.get("depth_loss", "plain") != "plain" or args.get("aux_l2", 0):
        p.error("use a control checkpoint with plain masked L1 and no existing moment adapter")
    dm = importlib.import_module(dm_name)
    args["value_moments"] = True
    model, mode, nv, kind, poses = build(args, dm)
    if not isinstance(model, OAAv2Depth) or kind != "spec":
        p.error("probe only supports the OAA spectrogram model")
    loaded = model.load_state_dict(ck["state_dict"], strict=False)
    if loaded.missing_keys != ["ray_mic.moment_scale"] or loaded.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    del ck
    model.to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    model.ray_mic.moment_scale.requires_grad_(True)

    train, val = dm.RotSet("train", mode), dm.RotSet("val", mode)
    if not len(train) or not len(val):
        p.error("empty train/val dataset; check the dataset root")
    if {s for s, _ in train.samples} & {s for s, _ in val.samples}:
        raise ValueError("Train and validation scenes overlap")
    ti = balanced_indices(train.samples, a.train_samples, a.seed)
    vi = balanced_indices(val.samples, a.val_samples, a.seed + 1)
    tr = DataLoader(Subset(train, ti), a.batch_size, shuffle=True, num_workers=a.workers,
                    generator=torch.Generator().manual_seed(a.seed))
    va = DataLoader(Subset(val, vi), a.batch_size, shuffle=False, num_workers=a.workers)
    max_depth = float(args.get("max_depth", 10.))
    if max_depth != float(dm.MAX_DEPTH):
        raise ValueError("Checkpoint and data-module depth normalisation disagree")
    a.out_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Explicit baseline: bypass the new readout, using exactly the same shared weights.
    model.ray_mic.value_moments = False
    before = score(model, va, device, poses, nv, max_depth, val.samples)
    model.ray_mic.value_moments = True
    # Identity check on actual input. Float64 bit-parity is separately enforced by CPU tests.
    first = next(iter(va))["spec"][:, :nv].to(device)
    with torch.no_grad(), amp(device):
        on = model(first, view_poses=poses)
        model.ray_mic.value_moments = False
        off = model(first, view_poses=poses)
        model.ray_mic.value_moments = True
    initial_diff = (on - off).abs().max().item()
    if initial_diff != 0:
        raise RuntimeError(f"Initial identity failed on real input: maxdiff={initial_diff}")
    del first, on, off
    # One fixed, explicitly restricted probe. Use the control's LR, no LR search,
    # no EMA, no validation selection; only the final adapter is scored.
    opt = torch.optim.AdamW([model.ray_mic.moment_scale], lr=float(args["lr"]),
                            weight_decay=float(args.get("wd", 1e-4)))
    wlat = cos_lat(256, device).view(1, 1, 256, 1)
    hist = []
    for epoch in range(a.passes):
        total = steps = 0
        for b in tr:
            opt.zero_grad(set_to_none=True)
            with amp(device):
                pred = model(b["spec"][:, :nv].to(device), view_poses=poses)
            loss = depth_loss(pred, b["depth"].to(device), b["mask"].to(device), "plain", wlat)
            if loss is None or not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite/empty training batch in probe")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.ray_mic.moment_scale], 1., error_if_nonfinite=True)
            opt.step()
            total += loss.item(); steps += 1
        hist.append({"pass": epoch + 1, "train_loss": total / steps, "steps": steps})
        print(json.dumps(hist[-1]), flush=True)
    after = score(model, va, device, poses, nv, max_depth, val.samples)
    report = {"kind": "frozen_backbone_train_only_adapter_probe", "not_a_final_result": True,
              "baseline_run": str(a.baseline_run), "baseline_args": dict(args, value_moments=False),
              "probe_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "initial_maxdiff": initial_diff, "train_indices": ti, "val_indices": vi,
              "train_samples": [train.samples[i] for i in ti], "val_samples": [val.samples[i] for i in vi],
              "before": before, "after": after, "history": hist,
              "metric_delta_after_minus_before": {k: after["metrics"][k] - v
                                                   for k, v in before["metrics"].items()},
              "moment_scale_absmean": model.ray_mic.moment_scale.detach().abs().mean().item(),
              "elapsed_seconds": time.monotonic() - start,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "torch": torch.__version__}
    (a.out_dir / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"moment_scale": model.ray_mic.moment_scale.detach().cpu(),
                "note": "Probe adapter only; requires the exact baseline checkpoint"}, a.out_dir / "adapter.pth")
    print(json.dumps(report["metric_delta_after_minus_before"], indent=2), flush=True)


if __name__ == "__main__":
    main()
