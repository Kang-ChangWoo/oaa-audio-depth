"""OAA trainer.

Recipe:
  * masked L1 depth loss only (depth / max_depth in [0,1], valid-pixel mask)
  * AdamW (wd 1e-4), linear warmup + cosine decay, grad-clip 1.0, bf16 autocast
  * EMA 0.999 of the weights (the EMA weights are what is saved and evaluated)
  * optional observation-token masking (--subset-aug / --vdrop-*) for nviews > 4
  * model selection on val (cos-latitude-weighted MAE); test numbers come from eval.py

Run (Replica 8 obs.):  DATA_MODULE=data_0422 python train_oaa.py --run-name oaa_r8 --nviews 8 \
                         --lr 5e-4 --warmup-ep 4 --epochs 40 --batch-size 3 --accum 11 --subset-aug --vdrop-kmax 4
See configs/ for the exact per-run settings behind every reported number.
"""
import os, json, math, time, argparse, copy, importlib, random
import torch.distributed as dist
import numpy as np
import torch

# data module selectable at runtime: DATA_MODULE=data_mp3d (Matterport3D, default) | data_0422 (Replica)
_DM = importlib.import_module(os.environ.get("DATA_MODULE", "data_mp3d"))
loader = _DM.loader
from model.oaa import OAAv2Depth


def cos_lat(h, device):
    v = torch.arange(h, device=device, dtype=torch.float32)
    return torch.cos((math.pi / 2) - (v + 0.5) / h * math.pi).clamp(min=1e-3)


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
    p.add_argument("--nviews", type=int, default=4, choices=[2, 4, 6, 8])
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--rounds", type=int, default=2)     # alternating intra/inter attention rounds
    p.add_argument("--lift-h", type=int, default=16)    # panoramic query grid (elevation x azimuth); must divide 256/512
    p.add_argument("--lift-w", type=int, default=32)
    p.add_argument("--stem-stride1", action="store_true")   # stride-1 stem on the native input (no 2x upsample)
    p.add_argument("--data-mode", default="")   # loader channel mode override (default derived from nviews)
    p.add_argument("--accum", type=int, default=1)   # grad-accumulation steps: effective batch = batch-size*accum
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    # observation-token masking (used for nviews > 4)
    p.add_argument("--subset-aug", action="store_true")   # zero out random observations per batch (poses fixed)
    p.add_argument("--vdrop-p", type=float, default=0.5)   # probability per batch
    p.add_argument("--vdrop-kmax", type=int, default=4)    # max observations zeroed
    p.add_argument("--vdrop-kstep", type=int, default=2)   # 2 = even counts only; 1 = any count
    p.add_argument("--vdrop-start", type=int, default=0)   # curriculum: masking off before this epoch
    p.add_argument("--vdrop-ramp", type=int, default=1)    # epochs to ramp p up to vdrop-p after start
    p.add_argument("--resume", default="")                 # last.pth: restore raw/EMA/opt and continue to --epochs
    # input-cue ablations (Table: input cues)
    p.add_argument("--pose-blind", action="store_true")   # same pose (0,+1) for every observation
    p.add_argument("--ear-blind", action="store_true")     # keep yaw, drop ear sign (all +1)
    p.add_argument("--yaw-flip", action="store_true")      # negate label yaws (dataset yaw-sign convention)
    # module ablations (Table: components), trained from scratch
    p.add_argument("--no-pose-emb", action="store_true")
    p.add_argument("--no-ray-emb", action="store_true")
    p.add_argument("--no-geo-bias", action="store_true")
    p.add_argument("--no-tf-pe", action="store_true")
    p.add_argument("--no-cross", action="store_true")      # baseline (a): no ray-observation cross-attention
    p.add_argument("--warmup-ep", type=float, default=4.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--out-dir", default="out")
    a = p.parse_args()
    # ckpt-args compatibility flags (eval.build reads them): this trainer always uses the full-resolution
    # multi-scale model with AdaLN conditioning and passes rounds/lift to the model.
    a.full_res = a.full_res_enc = a.multi_scale_lift = a.dec_deep = a.rounds_wired = True; a.cond_mode = "adaln"
    a.data_module = os.environ.get("DATA_MODULE", "data_mp3d")   # recorded so eval.py can refuse a dataset mismatch
    # DDP: auto-enabled under torchrun --nproc_per_node=N. Effective batch = batch-size(per-GPU) x world x accum.
    ddp = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if ddp else (0, 1)
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    is_main = rank == 0
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if ddp else torch.device("cuda")
    rd = os.path.join(a.out_dir, a.run_name)
    if is_main:
        os.makedirs(rd, exist_ok=True)

    dmode = a.data_mode or {2: "r2", 4: "cB", 6: "r6", 8: "r8"}[a.nviews]
    _ld = loader
    if ddp:
        from torch.utils.data import DataLoader, distributed as tdist
        tr_set = _DM.RotSet("train", dmode)
        tr_sampler = tdist.DistributedSampler(tr_set, num_replicas=world, rank=rank, shuffle=True, seed=a.seed)
        tr = DataLoader(tr_set, batch_size=a.batch_size, sampler=tr_sampler,
                        num_workers=a.num_workers, drop_last=True, pin_memory=True)
    else:
        tr = _ld("train", a.batch_size, True, a.num_workers, dmode)
    # val batch follows the train micro-batch (capped at 32): with full_res_enc at 6/8ch a bs-32
    # val forward spikes 6-8 GB above the training footprint and OOMs the epoch-0 validation.
    va = _ld("val", min(32, max(a.batch_size, 4)), False, a.num_workers, dmode)
    vp = getattr(_DM, "POSES", {}).get(dmode)   # OAA view_poses for this mode (None -> model default)
    if a.yaw_flip and vp:
        # dataset yaw convention (locations.json: +90 deg about +y, habitat -z forward = physical left) is
        # opposite to the model ERP convention (+z front, az +90 = +x right): negate label yaws
        vp = [(-y, e) for (y, e) in vp]
    if a.pose_blind:
        vp = [(0.0, 1.0)] * a.nviews             # same capacity, pose information removed
    elif a.ear_blind:
        vp = [(y, 1.0) for (y, e) in (vp or [])] or None

    model = OAAv2Depth(C=a.dim, nviews=a.nviews, rounds=a.rounds, lh=a.lift_h, lw=a.lift_w,
                       stem_stride1=a.stem_stride1, max_depth=a.max_depth,
                       no_pose_emb=a.no_pose_emb, no_ray_emb=a.no_ray_emb, no_geo_bias=a.no_geo_bias,
                       no_tf_pe=a.no_tf_pe, no_cross=a.no_cross).to(device)
    if is_main:
        print(f"[cfg] {vars(a)} params={sum(x.numel() for x in model.parameters())/1e6:.2f}M ddp_world={world}", flush=True)
    raw_model = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])],
                                                          find_unused_parameters=True)  # aux_head is not in the loss

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    accum = max(1, a.accum)
    steps_per_ep = math.ceil(len(tr) / accum)        # optimizer steps (schedule counts these, not micro-batches)
    total = a.epochs * steps_per_ep; warm = max(1, int(a.warmup_ep * steps_per_ep))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    wlat = cos_lat(256, device).view(1, 1, 256, 1)
    ema = copy.deepcopy(raw_model)
    for q in ema.parameters():
        q.requires_grad_(False)

    start_ep = 0
    if a.resume:                                     # continue training: restore model/EMA/opt, fast-forward the schedule
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ck["raw_state"]); ema.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        start_ep = int(ck.get("next_epoch", 0))
        assert start_ep < a.epochs, f"--epochs {a.epochs} <= already-trained {start_ep}"
        for _ in range(start_ep * steps_per_ep):     # move to the current position on the (extended) cosine
            sched.step()
        if is_main:
            print(f"[resume] {a.resume} ep{start_ep}-> lr={opt.param_groups[0]['lr']:.2e}", flush=True)

    best = 1e9; hist = []
    if a.resume:                                     # resuming in the same run dir: keep the existing best
        _td = os.path.join(os.path.dirname(a.resume), "train_done.json")
        if os.path.exists(_td) and os.path.dirname(a.resume) == rd:
            _j = json.load(open(_td)); best = _j.get("best_val_mae_m", 1e9); hist = _j.get("hist", [])
    nbatch = len(tr)
    for ep in range(start_ep, a.epochs):
        if ddp:
            tr_sampler.set_epoch(ep)
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        opt.zero_grad()
        for i, b in enumerate(tr):
            spec = b["spec"][:, :a.nviews].to(device, non_blocking=True)
            gt = b["depth"].to(device); mask = b["mask"].to(device)
            vp_in, spec_in = vp, spec
            _vp = a.vdrop_p * min(1.0, max(0.0, (ep - a.vdrop_start + 1) / max(1, a.vdrop_ramp)))
            if a.subset_aug and random.random() < _vp:          # observation-token masking (curriculum ramp)
                # never zero every observation: k < nviews
                _pool = [x for x in range(min(a.vdrop_kstep, 2), a.vdrop_kmax + 1, a.vdrop_kstep)
                         if x < spec_in.shape[1]] or [1]
                k = random.choice(_pool)
                idx = random.sample(range(spec_in.shape[1]), k)
                spec_in = spec_in.clone(); spec_in[:, idx] = 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                D = model(spec_in, view_poses=vp_in)
            loss = ((D.float() - gt).abs() * mask).sum() / mask.sum().clamp(min=1e-6)   # masked L1, nothing else
            (loss / accum).backward()                                  # accumulate grads over `accum` micro-batches
            run += float(loss.detach()); nb += 1
            if (i + 1) % accum == 0 or (i + 1) == nbatch:              # optimizer step per effective batch
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                with torch.no_grad():
                    for q, w in zip(ema.parameters(), raw_model.parameters()):
                        q.mul_(0.999).add_(w, alpha=0.001)
                    for q, w in zip(ema.buffers(), raw_model.buffers()):
                        q.copy_(w)
        run /= max(nb, 1)
        if is_main:
            vmae = quick_val(ema, va, device, a.max_depth, wlat, a.nviews, vp)
            hist.append({"epoch": ep, "loss": run, "val_mae_m": vmae})
            print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s loss={run:.4f} val_MAE={vmae:.4f}m", flush=True)
            if vmae < best:
                best = vmae
                torch.save({"state_dict": ema.state_dict(), "args": vars(a)}, os.path.join(rd, "best.pth"))
        if ddp:
            dist.barrier()
    if is_main:
        # last.pth = resumable bundle: EMA (state_dict, eval-compatible) + raw model + optimizer/scheduler/epoch
        torch.save({"state_dict": ema.state_dict(), "raw_state": raw_model.state_dict(),
                    "opt": opt.state_dict(), "sched_last_epoch": sched.last_epoch,
                    "next_epoch": a.epochs, "args": vars(a)}, os.path.join(rd, "last.pth"))
        json.dump({"best_val_mae_m": best, "hist": hist, "args": vars(a)},
                  open(os.path.join(rd, "train_done.json"), "w"), indent=2)
        print(f"[done] best val MAE={best:.4f}m -> {rd}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
