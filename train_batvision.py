"""BatVision baseline trainer (CNN recipe — note the LR differs from OAA's on purpose).

Recipe: masked L1, AdamW lr 2e-3 wd 1e-4, warmup 1 epoch + cosine, batch 32, 30 epochs, no EMA.
Run:  python3 train_batvision.py --run-name bat_r8_s0 --mode r8
"""
import os, json, math, time, argparse, importlib
import numpy as np
import torch

# data module selectable at runtime: DATA_MODULE=data (MP3D, default) | data_0422 (Replica)
_DM = importlib.import_module(os.environ.get("DATA_MODULE", "data"))
loader, IN_CH = _DM.loader, _DM.IN_CH
from model.batvision import RotDepth
from train_oaa import cos_lat, quick_val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--mode", default="cB", choices=["r2", "cB", "r6", "r8"])
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--feat-c", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup-ep", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--out-dir", default="out")
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda")
    rd = os.path.join(a.out_dir, a.run_name); os.makedirs(rd, exist_ok=True)

    tr = loader("train", a.batch_size, True, a.num_workers, a.mode)
    va = loader("val", 32, False, a.num_workers, a.mode)
    model = RotDepth(in_ch=IN_CH[a.mode], feat_c=a.feat_c, ngf=a.ngf).to(device)
    print(f"[cfg] {vars(a)} in_ch={IN_CH[a.mode]} params={sum(x.numel() for x in model.parameters())/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(tr); warm = max(1, int(a.warmup_ep * len(tr)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    wlat = cos_lat(256, device).view(1, 1, 256, 1)

    best = 1e9; hist = []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for b in tr:
            spec = b["spec"].to(device, non_blocking=True)
            gt = b["depth"].to(device); mask = b["mask"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                D = model(spec)
            loss = ((D.float() - gt).abs() * mask).sum() / mask.sum().clamp(min=1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += float(loss.detach()); nb += 1
        run /= max(nb, 1)
        vmae = quick_val(model, va, device, a.max_depth, wlat, IN_CH[a.mode])
        hist.append({"epoch": ep, "loss": run, "val_mae_m": vmae})
        print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s loss={run:.4f} val_MAE={vmae:.4f}m", flush=True)
        if vmae < best:
            best = vmae
            torch.save({"state_dict": model.state_dict(), "args": vars(a)}, os.path.join(rd, "best.pth"))
    torch.save({"state_dict": model.state_dict(), "args": vars(a)}, os.path.join(rd, "last.pth"))
    json.dump({"best_val_mae_m": best, "hist": hist, "args": vars(a)},
              open(os.path.join(rd, "train_done.json"), "w"), indent=2)
    print(f"[done] best val MAE={best:.4f}m -> {rd}", flush=True)


if __name__ == "__main__":
    main()
