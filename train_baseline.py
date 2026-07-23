"""Unified trainer for the comparison baselines (shared recipe = fair comparison to OAA/BatVision).

Recipe (same masked-L1 protocol as train_oaa/train_batvision): masked L1 only, AdamW + warmup +
cosine, bf16 autocast, grad-clip 1.0, model-select on cos-lat-weighted val MAE. Every baseline
predicts natively at 256x512 (no output upsampling), so the loss/metric are computed at 256x512
exactly like OAA.

Models (forward input differs):
  resnet / vit / beyond   consume the STFT `spec` (loader, mode r2/cB/r6/r8 -> in_ch 2/4/6/8)
  echoscan                consumes the raw binaural `wave` (wave_loader, 2ch @48k)
EchoDiffusion is trained separately in its isolated env (needs spec+wave+SD weights).

Run:
  python3 train_baseline.py --model resnet   --run-name rn_r8 --mode r8
  python3 train_baseline.py --model vit      --run-name vit_r2 --mode r2
  python3 train_baseline.py --model beyond   --run-name byd_r4 --mode cB
  python3 train_baseline.py --model echoscan --run-name es_s0
"""
import os, json, math, time, argparse, importlib
import numpy as np
import torch

# data module selectable at runtime: DATA_MODULE=data (MP3D, default) | data_0422 (Replica)
_DM = importlib.import_module(os.environ.get("DATA_MODULE", "data"))
loader, wave_loader, IN_CH = _DM.loader, _DM.wave_loader, _DM.IN_CH
from model import PretrainedResNet, PretrainedViT, BeyondI2DDepth, EchoScanDepth
from train_oaa import cos_lat

SPEC_MODELS = {"resnet": PretrainedResNet, "vit": PretrainedViT, "beyond": BeyondI2DDepth}
WAVE_MODELS = {"echoscan": EchoScanDepth}


def build_model(name, in_ch):
    if name == "resnet":
        return PretrainedResNet(in_ch=in_ch, pretrained=True)
    if name == "vit":
        return PretrainedViT(in_ch=in_ch, pretrained=True)
    if name == "beyond":
        return BeyondI2DDepth(in_ch=in_ch, pretrained_material=True)
    if name == "echoscan":
        return EchoScanDepth(in_ch=in_ch, fs=48000)
    raise ValueError(name)


def _key(name):
    return "wave" if name in WAVE_MODELS else "spec"


def param_report(model):
    """Return (total_M, trainable_M) and print a per-top-level-module parameter breakdown."""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[params] total={total/1e6:.2f}M  trainable={train/1e6:.2f}M  frozen={(total-train)/1e6:.2f}M", flush=True)
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if n:
            print(f"          - {name:16s} {n/1e6:7.2f}M", flush=True)
    return total / 1e6, train / 1e6


@torch.no_grad()
def quick_val(model, va, device, max_depth, wlat, name, in_ch):
    model.eval(); tot = wn = 0.0
    k = _key(name)
    for b in va:
        x = b[k] if k == "wave" else b[k][:, :in_ch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            D = model(x.to(device)).float() * max_depth
        gt = b["depth"].to(device) * max_depth
        w = wlat * b["mask"].to(device)
        tot += ((D - gt).abs() * w).sum().item(); wn += w.sum().item()
    return tot / max(wn, 1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(SPEC_MODELS) + list(WAVE_MODELS))
    p.add_argument("--run-name", required=True)
    p.add_argument("--mode", default="r2")   # loader mode (data.py: r2/cB/r6/r8; data_0422: r2/fb)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-ep", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--out-dir", default="comparison")   # all baseline results live under comparison/
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda")
    rd = os.path.join(a.out_dir, a.run_name); os.makedirs(rd, exist_ok=True)

    is_wave = a.model in WAVE_MODELS
    in_ch = IN_CH[a.mode]
    if is_wave:
        tr = wave_loader("train", a.batch_size, True, a.num_workers, a.mode)
        va = wave_loader("val", 32, False, a.num_workers, a.mode)
    else:
        tr = loader("train", a.batch_size, True, a.num_workers, a.mode)
        va = loader("val", 32, False, a.num_workers, a.mode)

    model = build_model(a.model, in_ch).to(device)
    k = _key(a.model)
    cfg = dict(vars(a)); cfg["in_ch"] = in_ch
    print(f"[cfg] {cfg}", flush=True)
    tot_m, tr_m = param_report(model)
    cfg["params_M"], cfg["trainable_M"] = round(tot_m, 3), round(tr_m, 3)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(tr); warm = max(1, int(a.warmup_ep * len(tr)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    wlat = cos_lat(256, device).view(1, 1, 256, 1)

    best = 1e9; hist = []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for b in tr:
            x = b[k] if is_wave else b[k][:, :in_ch]
            x = x.to(device, non_blocking=True)
            gt = b["depth"].to(device); mask = b["mask"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                D = model(x)
            loss = ((D.float() - gt).abs() * mask).sum() / mask.sum().clamp(min=1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += float(loss.detach()); nb += 1
        run /= max(nb, 1)
        vmae = quick_val(model, va, device, a.max_depth, wlat, a.model, in_ch)
        hist.append({"epoch": ep, "loss": run, "val_mae_m": vmae})
        print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s loss={run:.4f} val_MAE={vmae:.4f}m", flush=True)
        if vmae < best:
            best = vmae
            torch.save({"state_dict": model.state_dict(), "args": cfg}, os.path.join(rd, "best.pth"))
    torch.save({"state_dict": model.state_dict(), "args": cfg}, os.path.join(rd, "last.pth"))
    json.dump({"best_val_mae_m": best, "hist": hist, "args": cfg},
              open(os.path.join(rd, "train_done.json"), "w"), indent=2)
    print(f"[done] best val MAE={best:.4f}m -> {rd}", flush=True)


if __name__ == "__main__":
    main()
