"""OAA baseline trainer — the exact champion recipe, nothing else.

Recipe (decisive parts):
  * masked L1 only (silog/berhu/bins/aux all tested and rejected on 3-seed test)
  * AdamW lr 1e-3, wd 1e-4, warmup 4 epochs + cosine  (transformer LR; 2e-3 destabilises.
    NOTE: at nviews=8, lr 1e-3 is borderline — 1 of 3 seeds failed to train; if the loss
    rebounds right after warmup and never recovers, restart or use --lr 5e-4)
  * EMA 0.999 (the saved weights), bf16 autocast, grad-clip 1.0, batch 32, 30 epochs
  * model-select on val (cos-latitude-weighted MAE), report on test — val runs ~0.11 higher

Run:  python3 train_oaa.py --run-name r8_s0 --nviews 8
"""
import os, json, math, time, argparse, copy
import numpy as np
import torch

from data import loader
from oaa import OAAv2Depth


def cos_lat(h, device):
    v = torch.arange(h, device=device, dtype=torch.float32)
    return torch.cos((math.pi / 2) - (v + 0.5) / h * math.pi).clamp(min=1e-3)


@torch.no_grad()
def quick_val(model, va, device, max_depth, wlat, nv):
    model.eval(); tot = wn = 0.0
    for b in va:
        sp = b["spec"][:, :nv]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            D = model(sp.to(device)).float() * max_depth
        gt = b["depth"].to(device) * max_depth
        w = wlat * b["mask"].to(device)
        tot += ((D - gt).abs() * w).sum().item(); wn += w.sum().item()
    return tot / max(wn, 1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--nviews", type=int, default=4, choices=[2, 4, 6, 8])
    p.add_argument("--cond-mode", default="adaln", choices=["add", "adaln"])
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-ep", type=float, default=4.0)
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

    dmode = {2: "r2", 4: "cB", 6: "r6", 8: "r8"}[a.nviews]
    tr = loader("train", a.batch_size, True, a.num_workers, dmode)
    va = loader("val", 32, False, a.num_workers, dmode)

    model = OAAv2Depth(C=a.dim, nviews=a.nviews, cond_mode=a.cond_mode, max_depth=a.max_depth).to(device)
    print(f"[cfg] {vars(a)} params={sum(x.numel() for x in model.parameters())/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(tr); warm = max(1, int(a.warmup_ep * len(tr)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    wlat = cos_lat(256, device).view(1, 1, 256, 1)
    ema = copy.deepcopy(model)
    for q in ema.parameters():
        q.requires_grad_(False)

    best = 1e9; hist = []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for b in tr:
            spec = b["spec"][:, :a.nviews].to(device, non_blocking=True)
            gt = b["depth"].to(device); mask = b["mask"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                D = model(spec)
            loss = ((D.float() - gt).abs() * mask).sum() / mask.sum().clamp(min=1e-6)   # masked L1, nothing else
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                for q, w in zip(ema.parameters(), model.parameters()):
                    q.mul_(0.999).add_(w, alpha=0.001)
                for q, w in zip(ema.buffers(), model.buffers()):
                    q.copy_(w)
            run += float(loss.detach()); nb += 1
        run /= max(nb, 1)
        vmae = quick_val(ema, va, device, a.max_depth, wlat, a.nviews)
        hist.append({"epoch": ep, "loss": run, "val_mae_m": vmae})
        print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s loss={run:.4f} val_MAE={vmae:.4f}m", flush=True)
        if vmae < best:
            best = vmae
            torch.save({"state_dict": ema.state_dict(), "args": vars(a)}, os.path.join(rd, "best.pth"))
    torch.save({"state_dict": ema.state_dict(), "args": vars(a)}, os.path.join(rd, "last.pth"))
    json.dump({"best_val_mae_m": best, "hist": hist, "args": vars(a)},
              open(os.path.join(rd, "train_done.json"), "w"), indent=2)
    print(f"[done] best val MAE={best:.4f}m -> {rd}", flush=True)


if __name__ == "__main__":
    main()
