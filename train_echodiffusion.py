"""EchoDiffusion baseline trainer — runs in the ISOLATED env (needs ldm/SD-UNet/wav2vec2).

  conda activate <echodiff_env>   # torch 1.13.1+cu117, mmcv 1.7.1, ...
  CUDA_VISIBLE_DEVICES=5 DATA_MODULE=data_0422 R0422_SPLIT=off3 \
    <echodiff_env>/bin/python train_echodiffusion.py --run-name eco_r2

EchoDiffusion consumes BOTH the STFT spec and the raw waveform, so it uses data_0422's
spec_wave_loader. Same masked-L1 recipe / cos-lat val as the other baselines; results saved under
comparison/ so eval.py picks them up. Lower LR (1e-4) and small batch (SD-UNet is heavy).
"""
import os, json, math, time, argparse, random
from contextlib import nullcontext
import numpy as np
import torch

from model.echodiffusion import EchoDiffusionDepth

from core.data import get_data_module
_DM = get_data_module("data_0422")

# fp32 by default: torch 1.13 bf16 autocast dies in ASPP's bilinear upsample
# ("upsample_bilinear2d not implemented for BFloat16") and the checkpointed SD-UNet
# attention has the same dtype clash that killed the fp16 attempt. ECO_BF16=1 to opt in.
_BF16 = os.environ.get("ECO_BF16", "") == "1"


def _amp():
    return torch.autocast("cuda", dtype=torch.bfloat16) if _BF16 else nullcontext()


from core.metrics import cos_lat


@torch.no_grad()
def quick_val(model, va, device, max_depth, wlat, wch=2, cide=None):
    model.eval(); tot = wn = 0.0
    for b in va:
        ce = cide[b["idx"]].to(device) if cide is not None else None
        with _amp():
            D = model(b["spec"].to(device), b["wave"][:, :wch].to(device), cide=ce)
        D = D.float() * max_depth
        gt = b["depth"].to(device) * max_depth
        w = wlat * b["mask"].to(device)
        tot += ((D - gt).abs() * w).sum().item(); wn += w.sum().item()
    return tot / max(wn, 1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--mode", default="r2")   # data mode: r2/fb/r6/r8 -> spec in_ch 2/4/6/8
    # waveform-branch ablation (2026-07-23): std = original CIDE on the front 2ch (fixed even as
    # spec channels grow); all = CIDE gets ALL of the mode's wave channels (2/4/6/8 — the channel-
    # scaling variant the user asked for); long = 2ch but longer raw cut; none = CIDE removed
    p.add_argument("--wave-mode", default="std", choices=["std", "all", "long", "none"])
    p.add_argument("--chdrop", action="store_true")        # same channel drop as OAA fin (p=0.5, k in {2,4}); dropped front channels also zero the CIDE waveform
    p.add_argument("--chdrop-p", type=float, default=0.5)
    p.add_argument("--chdrop-kmax", type=int, default=4)
    p.add_argument("--wave-window", type=int, default=48000)  # samples for --wave-mode long (1.0 s @48k)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ep", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=12)  # early stop: epochs without val improvement (eco always peaks ep14-16 then degrades monotonically; best.pth already saved)
    p.add_argument("--port", default="faithful", choices=["faithful", "enhanced"])
    p.add_argument("--cide-cache", default="")   # cache_cide/*.npy — precomputed wav2vec2 embeddings (tools/build_cide_cache.py)  # faithful=as in the original (128x128 + post-hoc upsample), enhanced=improved port (earlier round)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--out-dir", default="comparison")
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda")
    rd = os.path.join(a.out_dir, a.run_name); os.makedirs(rd, exist_ok=True)

    in_ch = _DM.IN_CH[a.mode]
    ww = {"wave_window": a.wave_window} if a.wave_mode == "long" else {}
    tr = _DM.spec_wave_loader("train", a.batch_size, True, a.num_workers, a.mode, **ww)
    va = _DM.spec_wave_loader("val", 12, False, a.num_workers, a.mode, **ww)
    wave_ch = in_ch if a.wave_mode == "all" else 2
    model = EchoDiffusionDepth(in_ch=in_ch, wave_mode="none" if a.wave_mode == "none" else "cide",
                               wave_ch=wave_ch, faithful=(a.port == "faithful")).to(device)
    cfg = dict(vars(a)); cfg["model"] = "echodiffusion"; cfg["amp"] = "bf16" if _BF16 else "fp32"
    tot = sum(x.numel() for x in model.parameters()); trn = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"[cfg] {cfg}", flush=True)
    print(f"[params] total={tot/1e6:.2f}M trainable={trn/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(tr); warm = max(1, int(a.warmup_ep * len(tr)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    wlat = cos_lat(256, device).view(1, 1, 256, 1)

    cide_all = None
    if a.cide_cache:
        cide_all = {sp: torch.from_numpy(np.load(a.cide_cache.replace("SPLIT", sp))).float()
                    for sp in ("train", "val")}
        print(f"[cide-cache] loaded {a.cide_cache}", flush=True)

    best = 1e9; best_ep = -1; hist = []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for b in tr:
            spec = b["spec"].to(device, non_blocking=True); wave = b["wave"][:, :wave_ch].to(device, non_blocking=True)
            gt = b["depth"].to(device); mask = b["mask"].to(device)
            if a.chdrop and random.random() < a.chdrop_p:      # mirrors OAA vdrop: zero k channels per batch (no masking at val)
                _pool = [x for x in range(2, a.chdrop_kmax + 1, 2) if x < spec.shape[1]] or [1]
                _idx = random.sample(range(spec.shape[1]), random.choice(_pool))
                spec = spec.clone(); spec[:, _idx] = 0
                _dead = [j for j in _idx if j < wave.shape[1]]
                if _dead:
                    wave = wave.clone(); wave[:, _dead] = 0
            ce = cide_all["train"][b["idx"]].to(device) if cide_all else None
            with _amp():   # CIDE gets wave_ch channels (std:2, all:mode)
                D = model(spec, wave, cide=ce)
            loss = ((D.float() - gt).abs() * mask).sum() / mask.sum().clamp(min=1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += float(loss.detach()); nb += 1
        run /= max(nb, 1)
        vmae = quick_val(model, va, device, a.max_depth, wlat, wave_ch,
                         cide_all["val"] if cide_all else None)
        hist.append({"epoch": ep, "loss": run, "val_mae_m": vmae})
        print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s loss={run:.4f} val_MAE={vmae:.4f}m", flush=True)
        if vmae < best:
            best = vmae; best_ep = ep
            torch.save({"state_dict": model.state_dict(), "args": cfg}, os.path.join(rd, "best.pth"))
        elif ep - best_ep >= a.patience:
            print(f"[early-stop] no val improvement since ep {best_ep} (patience {a.patience})", flush=True)
            break
    torch.save({"state_dict": model.state_dict(), "args": cfg}, os.path.join(rd, "last.pth"))
    json.dump({"best_val_mae_m": best, "hist": hist, "args": cfg},
              open(os.path.join(rd, "train_done.json"), "w"), indent=2)
    print(f"[done] best val MAE={best:.4f}m -> {rd}", flush=True)


if __name__ == "__main__":
    main()
