"""Test-set evaluation of one run directory (base-env models)."""
import os
import torch

from core.ckpt import load_run
from core.metrics import MetricAccumulator
from core import pairfeat


@torch.no_grad()
def evaluate(run_dir, DM, ckpt="best", device="cuda", max_depth=10.0, batch_size=None):
    """-> dict of metrics (core.metrics.KEYS + bands + 'Params(M)'). batch_size defaults to $EVAL_BS or 32."""
    model, dmode, nch, kind, poses, ck = load_run(run_dir, DM, ckpt, device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    max_depth = ck["args"].get("max_depth", max_depth)
    # --pair-feat (2026-10-27): an OAA cell trained with pair input planes must be scored with the
    # same planes built from the loader's magnitude spec; absent in every earlier checkpoint ->
    # "none" -> the forward below is statement-identical to the reference.
    pf = (ck["args"].get("pair_feat") or "none") if kind == "spec" else "none"
    pf_chans = list(DM._CH[dmode][:nch]) if pf != "none" else None
    bs = batch_size or int(os.environ.get("EVAL_BS", "32"))   # lower (EVAL_BS=4) to fit a shared/contended GPU
    ld = DM.wave_loader("test", bs, False, 5, dmode) if kind == "wave" else DM.loader("test", bs, False, 5, dmode)
    acc = MetricAccumulator(device)
    for b in ld:
        x = b["wave"][:, :nch].to(device) if kind == "wave" else b["spec"][:, :nch].to(device)
        if pf != "none":
            x = pairfeat.build(x, pf_chans, pf)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(torch.device(device).type == "cuda")):
            D = (model(x, view_poses=poses) if poses is not None else model(x)).float() * max_depth
        acc.update(D, b["depth"].to(device) * max_depth, b["mask"].to(device))
    out = acc.result(); out["Params(M)"] = params_m
    return out
