"""0820 experiment trainer: OAA with a pretrained Audio Foundation Model per-observation encoder.

Identical recipe to train_oaa.py (masked L1, AdamW wd 1e-4, warmup+cosine, grad-clip 1.0, bf16
autocast, EMA 0.999, val-MAE model selection) with exactly two changes:
  * --audio-backbone {cnn,audiomosaic,bat,eat,sslam,m2d,m2d_plain}: swaps the coarse per-observation
    encoder for a pretrained ViT-B/16 AFM (model/audio_backbones_0820.py). "cnn" = original model.
  * two optimizer groups: pretrained AFM tensors at --afm-lr-ratio x lr (default 0.1), all new /
    OAA / decoder params at the base lr. Same policy for every AFM and both datasets.

Screening run (4 obs., Replica):
  DATA_MODULE=data_0422 python 0820_train_oaa_afm.py --run-name 0820_bat_fb --audio-backbone bat \
      --nviews 4 --data-mode fb --lr 5e-4 --warmup-ep 4 --epochs 40 --batch-size 8 --accum 4 \
      --out-dir comparison_0820
--afm-random-init keeps the identical architecture/LR groups but skips the pretrained weights
(Stage-2 pretraining ablation). Pretrained checkpoint loading failures are fatal by design.
"""
import os, json, math, time, argparse, copy, random
import numpy as np
import torch

from core.data import get_data_module
from core.metrics import cos_lat
from model.oaa import OAAv2Depth
from model.audio_backbones_0820 import BACKBONES, build_afm_model, make_param_groups

_DM = get_data_module()
loader = _DM.loader


@torch.no_grad()
def quick_val(model, va, device, max_depth, wlat, nv, vp=None):
    model.eval(); tot = wn = 0.0
    for b in va:
        sp = b["spec"][:, :nv].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            D = model(sp, view_poses=vp).float() * max_depth
        gt = b["depth"].to(device) * max_depth
        w = wlat * b["mask"].to(device)
        tot += ((D - gt).abs() * w).sum().item(); wn += w.sum().item()
    return tot / max(wn, 1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--audio-backbone", required=True, choices=list(BACKBONES))
    p.add_argument("--afm-lr-ratio", type=float, default=0.1)   # pretrained-AFM LR = ratio * lr
    p.add_argument("--afm-random-init", action="store_true")    # Stage-2 ablation: same arch, no pretrained init
    p.add_argument("--nviews", type=int, default=4, choices=[2, 4, 6, 8])
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--lift-h", type=int, default=16)
    p.add_argument("--lift-w", type=int, default=32)
    p.add_argument("--stem-stride1", action="store_true")
    p.add_argument("--data-mode", default="")
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--subset-aug", action="store_true")
    p.add_argument("--vdrop-p", type=float, default=0.5)
    p.add_argument("--vdrop-kmax", type=int, default=4)
    p.add_argument("--vdrop-kstep", type=int, default=2)
    p.add_argument("--vdrop-start", type=int, default=0)
    p.add_argument("--vdrop-ramp", type=int, default=1)
    p.add_argument("--resume", default="")
    p.add_argument("--warmup-ep", type=float, default=4.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--out-dir", default="comparison_0820")
    a = p.parse_args()
    # ckpt-args compatibility (core.ckpt.build / eval.py read these)
    a.full_res = a.full_res_enc = a.multi_scale_lift = a.dec_deep = a.rounds_wired = True; a.cond_mode = "adaln"
    a.data_module = os.environ.get("DATA_MODULE", "data_mp3d")
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = torch.device("cuda")
    rd = os.path.join(a.out_dir, a.run_name); os.makedirs(rd, exist_ok=True)

    dmode = a.data_mode or {2: "r2", 4: "cB", 6: "r6", 8: "r8"}[a.nviews]
    tr = loader("train", a.batch_size, True, a.num_workers, dmode)
    va = loader("val", min(32, max(a.batch_size, 4)), False, a.num_workers, dmode)
    vp = getattr(_DM, "POSES", {}).get(dmode)

    if a.audio_backbone == "cnn":
        model = OAAv2Depth(C=a.dim, nviews=a.nviews, rounds=a.rounds, lh=a.lift_h, lw=a.lift_w,
                           stem_stride1=a.stem_stride1, max_depth=a.max_depth).to(device)
        a.afm_checkpoint = "none (original CNN encoder)"
        groups = [{"params": list(model.parameters()), "lr": a.lr, "weight_decay": a.wd}]
    else:
        model = build_afm_model(vars(a), pretrained=not a.afm_random_init).to(device)
        a.afm_checkpoint = model.enc.afm.checkpoint_id
        assert a.afm_random_init or model.enc.afm.pretrained_loaded, "pretrained AFM weights not loaded"
        groups = make_param_groups(model, a.lr, a.afm_lr_ratio, a.wd)
        n_pre = sum(p.numel() for p in groups[0]["params"])
        print(f"[cfg] Audio backbone: {a.audio_backbone} | ckpt: {a.afm_checkpoint} | "
              f"pretrained loaded: {'NO (random init ablation)' if a.afm_random_init else 'YES'} | "
              f"AFM dim 768 -> OAA dim {a.dim} | target grid {a.lift_h}x{a.lift_w} | "
              f"AFM LR {a.lr*a.afm_lr_ratio:.1e} ({n_pre/1e6:.1f}M params) / base LR {a.lr:.1e} | "
              f"trainable AFM: YES, full fine-tuning", flush=True)
    nparam = sum(x.numel() for x in model.parameters())
    print(f"[cfg] {vars(a)} params={nparam/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(groups)
    accum = max(1, a.accum)
    steps_per_ep = math.ceil(len(tr) / accum)
    total = a.epochs * steps_per_ep; warm = max(1, int(a.warmup_ep * steps_per_ep))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    wlat = cos_lat(256, device).view(1, 1, 256, 1)
    ema = copy.deepcopy(model)
    for q in ema.parameters():
        q.requires_grad_(False)

    start_ep = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["raw_state"]); ema.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        start_ep = int(ck.get("next_epoch", 0))
        assert start_ep < a.epochs, f"--epochs {a.epochs} <= already-trained {start_ep}"
        for _ in range(start_ep * steps_per_ep):
            sched.step()
        print(f"[resume] {a.resume} ep{start_ep}-> lr={opt.param_groups[-1]['lr']:.2e}", flush=True)

    best = 1e9; hist = []
    if a.resume:
        _td = os.path.join(os.path.dirname(a.resume), "train_done.json")
        if os.path.exists(_td) and os.path.dirname(a.resume) == rd:
            _j = json.load(open(_td)); best = _j.get("best_val_mae_m", 1e9); hist = _j.get("hist", [])
    nbatch = len(tr)
    for ep in range(start_ep, a.epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        opt.zero_grad()
        for i, b in enumerate(tr):
            spec = b["spec"][:, :a.nviews].to(device, non_blocking=True)
            gt = b["depth"].to(device); mask = b["mask"].to(device)
            spec_in = spec
            _vp = a.vdrop_p * min(1.0, max(0.0, (ep - a.vdrop_start + 1) / max(1, a.vdrop_ramp)))
            if a.subset_aug and random.random() < _vp:
                _pool = [x for x in range(min(a.vdrop_kstep, 2), a.vdrop_kmax + 1, a.vdrop_kstep)
                         if x < spec_in.shape[1]] or [1]
                k = random.choice(_pool)
                idx = random.sample(range(spec_in.shape[1]), k)
                spec_in = spec_in.clone(); spec_in[:, idx] = 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                D = model(spec_in, view_poses=vp)
            loss = ((D.float() - gt).abs() * mask).sum() / mask.sum().clamp(min=1e-6)
            (loss / accum).backward()
            run += float(loss.detach()); nb += 1
            if (i + 1) % accum == 0 or (i + 1) == nbatch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                with torch.no_grad():
                    for q, w in zip(ema.parameters(), model.parameters()):
                        q.mul_(0.999).add_(w, alpha=0.001)
                    for q, w in zip(ema.buffers(), model.buffers()):
                        q.copy_(w)
        run /= max(nb, 1)
        vmae = quick_val(ema, va, device, a.max_depth, wlat, a.nviews, vp)
        hist.append({"epoch": ep, "loss": run, "val_mae_m": vmae})
        print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s loss={run:.4f} val_MAE={vmae:.4f}m "
              f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
        if vmae < best:
            best = vmae
            torch.save({"state_dict": ema.state_dict(), "args": vars(a)}, os.path.join(rd, "best.pth"))
        torch.save({"state_dict": ema.state_dict(), "raw_state": model.state_dict(),
                    "opt": opt.state_dict(), "sched_last_epoch": sched.last_epoch,
                    "next_epoch": ep + 1, "args": vars(a)}, os.path.join(rd, "last.pth"))
    json.dump({"best_val_mae_m": best, "hist": hist, "args": vars(a)},
              open(os.path.join(rd, "train_done.json"), "w"), indent=2)
    print(f"[done] best val MAE={best:.4f}m -> {rd}", flush=True)


if __name__ == "__main__":
    main()
