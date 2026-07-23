"""EchoDiffusion baseline trainer — runs in the ISOLATED env (needs ldm/SD-UNet/wav2vec2).

  conda activate /root/local1/changwoo/echodiff_env   # torch 1.13.1+cu117, mmcv 1.7.1, ...
  CUDA_VISIBLE_DEVICES=5 DATA_MODULE=data_0422 R0422_SPLIT=off3 \
    /root/local1/changwoo/echodiff_env/bin/python train_echodiffusion.py --run-name eco_r2

EchoDiffusion consumes BOTH the STFT spec and the raw waveform, so it uses data_0422's
spec_wave_loader. Same masked-L1 recipe / cos-lat val as the other baselines; results saved under
comparison/ so eval.py picks them up. Lower LR (1e-4) and small batch (SD-UNet is heavy).
"""
import os, json, math, time, argparse, importlib
import numpy as np
import torch

from model.echodiffusion import EchoDiffusionDepth

_DM = importlib.import_module(os.environ.get("DATA_MODULE", "data_0422"))


def cos_lat(h, device):
    v = torch.arange(h, device=device, dtype=torch.float32)
    return torch.cos((math.pi / 2) - (v + 0.5) / h * math.pi).clamp(min=1e-3)


@torch.no_grad()
def quick_val(model, va, device, max_depth, wlat):
    model.eval(); tot = wn = 0.0
    for b in va:
        D = model(b["spec"].to(device), b["wave"][:, :2].to(device)).float() * max_depth   # fp32; CIDE uses front 2ch wave
        gt = b["depth"].to(device) * max_depth
        w = wlat * b["mask"].to(device)
        tot += ((D - gt).abs() * w).sum().item(); wn += w.sum().item()
    return tot / max(wn, 1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--mode", default="r2")   # data_0422 mode: r2/fb/r6/r8 -> spec in_ch 2/4/6/8
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ep", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--out-dir", default="comparison")
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda")
    rd = os.path.join(a.out_dir, a.run_name); os.makedirs(rd, exist_ok=True)

    in_ch = _DM.IN_CH[a.mode]
    tr = _DM.spec_wave_loader("train", a.batch_size, True, a.num_workers, a.mode)
    va = _DM.spec_wave_loader("val", 12, False, a.num_workers, a.mode)
    model = EchoDiffusionDepth(in_ch=in_ch).to(device)
    cfg = dict(vars(a)); cfg["model"] = "echodiffusion"
    tot = sum(x.numel() for x in model.parameters()); trn = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"[cfg] {cfg}", flush=True)
    print(f"[params] total={tot/1e6:.2f}M trainable={trn/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(tr); warm = max(1, int(a.warmup_ep * len(tr)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    wlat = cos_lat(256, device).view(1, 1, 256, 1)

    best = 1e9; hist = []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for b in tr:
            spec = b["spec"].to(device, non_blocking=True); wave = b["wave"][:, :2].to(device, non_blocking=True)
            gt = b["depth"].to(device); mask = b["mask"].to(device)
            D = model(spec, wave)                       # fp32; CIDE wav2vec2 branch takes front 2ch wave
            loss = ((D.float() - gt).abs() * mask).sum() / mask.sum().clamp(min=1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += float(loss.detach()); nb += 1
        run /= max(nb, 1)
        vmae = quick_val(model, va, device, a.max_depth, wlat)
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
