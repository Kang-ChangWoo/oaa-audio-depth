"""E144: train_oaa_e143.py + crash-safe checkpointing. No recipe change.

E143 saved last.pth exactly once, after the final epoch, so an 80-epoch run that died at epoch 60
could not be resumed at all. E144 writes the resumable bundle every epoch via a temp file +
os.replace (the HEAR360 EchoDiffusion pattern, oaa-audio-depth b7d2259), carries best/hist/RNG
inside it, and accepts --resume auto.

The three arms (ctrl / bin4 / invfreq) all run from THIS single file, so unlike E135 vs E114 there
is no cross-version confound between them. Guarantees inherited from E143 still hold:
--loss-weight none is train_oaa_e135.py's recipe exactly; --cue none --loss-weight invfreq is
train_oaa_e114.py's. A fresh (non-resumed) run is bit-identical to E143: the added calls are
torch.save plus RNG *getters*, none of which advance a generator.

E143 docstring follows.
"""
"""E143: E135's interaural input channels (--cue) + E114's distance-reweighted loss (--loss-weight).

Byte-for-byte train_oaa_e135.py with exactly ONE addition: the masked-L1 loss gains the per-range-bin
weight from train_oaa_e114.py. The two changes live in different layers and do not interact in code:
  * --cue   (E135) changes OAAv2Depth(in_ch=...) and the data module's input channels.
  * --loss-weight (E114) changes only the pixel weighting inside the masked L1.
With --loss-weight none this file is train_oaa_e135.py exactly; with --cue none and
--loss-weight invfreq it is train_oaa_e114.py's recipe. See PREREG_E143.md section 0.

E135 docstring follows.
"""
"""E135: OAA trainer with explicit interaural input channels (copy of hear360/train_oaa.py).

The ONLY change vs the original recipe is the number of input channels per ear observation
(--cue, wired into OAAv2Depth(in_ch=...) which train_oaa.py always left at 1). The channels
themselves are built by the data module data_0422_bin.py; see its docstring and PREREG_E135.md.
Model / optimizer / cosine schedule / EMA / masked-L1 loss / split are byte-for-byte the
original, so E133 `ctrl` remains a valid comparison arm.

Original docstring follows.


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
import os, json, math, time, argparse, copy, random
import torch.distributed as dist
import numpy as np
import torch

from core.data import get_data_module
from core.metrics import cos_lat
from model.oaa import OAAv2Depth

CUE_CH = {"none": 1, "ild": 2, "ipd": 3, "bin4": 4}   # channels per ear observation


# ---- E144: crash-safe checkpointing (the only change vs train_oaa_e143.py) ----
def atomic_save(obj, path):
    """Write to <path>.tmp, fsync, then os.replace: a kill mid-write cannot truncate <path>."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rng_state():
    # getters only -- these do not advance any generator, so fresh runs match E143 bit-for-bit.
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_state(s):
    torch.set_rng_state(s["torch"])
    try:
        torch.cuda.set_rng_state_all(s["cuda"])
    except Exception as e:                           # device count can differ across nodes
        print(f"[resume] cuda rng restore skipped: {e}", flush=True)
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def ckpt_bundle(ema, raw_model, opt, sched, next_epoch, best, hist, a):
    return {"state_dict": ema.state_dict(), "raw_state": raw_model.state_dict(),
            "opt": opt.state_dict(), "sched_last_epoch": sched.last_epoch,
            "next_epoch": next_epoch, "best_val_mae_m": best, "hist": hist,
            "rng": rng_state(), "args": vars(a)}


# ---- E114 loss reweighting, transplanted verbatim from train_oaa_e114.py ----
BIN_DIST_PATH = os.environ.get("E143_BIN_DIST", "/root/storage/e143_code/gate0_bin_dist.json")


def load_weight_fn(kind: str, device):
    if kind == "none":
        def w(d_m):
            return torch.ones_like(d_m)
        return w
    dist_j = json.load(open(BIN_DIST_PATH))
    edges = torch.tensor(dist_j["bin_edges"], device=device)          # (0.5, 1.5, 4.0)
    if kind == "invfreq":
        bw = torch.tensor(dist_j["invfreq_weight_train"], device=device)  # (4,)

        def w(d_m):
            idx = torch.bucketize(d_m, edges)   # 0:<0.5 1:0.5-1.5 2:1.5-4 3:>4
            return bw[idx]
        return w
    if kind == "sqrt":
        mean_sqrt = float(dist_j["mean_sqrt_d_train"])

        def w(d_m):
            return torch.sqrt(d_m.clamp(0.05, 10.0)) / mean_sqrt
        return w
    raise ValueError(kind)

_DM = get_data_module()          # DATA_MODULE=data_mp3d (Matterport3D, default) | data_0422 (Replica)
loader = _DM.loader


@torch.no_grad()
def quick_val(model, va, device, max_depth, wlat, nch, vp=None):
    model.eval(); tot = wn = 0.0
    for b in va:
        sp = b["spec"][:, :nch].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            D = model(sp, view_poses=vp).float() * max_depth
        gt = b["depth"].to(device) * max_depth
        w = wlat * b["mask"].to(device)
        tot += ((D - gt).abs() * w).sum().item(); wn += w.sum().item()
    return tot / max(wn, 1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--cue", default="none", choices=sorted(CUE_CH),
                   help="interaural input channels per ear observation (data_0422_bin.py)")
    p.add_argument("--loss-weight", default="none", choices=["none", "invfreq", "sqrt"],
                   help="E114 per-range-bin loss weighting; none = original masked L1")
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
    a.in_ch = CUE_CH[a.cue]
    _env_cue = os.environ.get("E135_CUE", "none")
    assert _env_cue == a.cue, f"--cue {a.cue} but E135_CUE={_env_cue!r}: the data module builds the channels"
    if a.in_ch > 1:
        assert os.environ.get("DATA_MODULE") == "data_0422_bin", "cue channels need DATA_MODULE=data_0422_bin"
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
    if a.resume == "auto":       # E144: relaunch-safe -- resume iff this run dir already has a last.pth
        _auto = os.path.join(rd, "last.pth")
        a.resume = _auto if os.path.exists(_auto) else ""
        print(f"[resume] auto -> {a.resume or 'none (fresh start)'}", flush=True)

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

    model = OAAv2Depth(C=a.dim, nviews=a.nviews, in_ch=a.in_ch, rounds=a.rounds, lh=a.lift_h, lw=a.lift_w,
                       stem_stride1=a.stem_stride1, max_depth=a.max_depth,
                       no_pose_emb=a.no_pose_emb, no_ray_emb=a.no_ray_emb, no_geo_bias=a.no_geo_bias,
                       no_tf_pe=a.no_tf_pe, no_cross=a.no_cross).to(device)
    if is_main:
        print(f"[E143] cue={a.cue} loss_weight={a.loss_weight} in_ch={a.in_ch} spec_ch={a.nviews * a.in_ch} "
              f"scale_ild={getattr(_DM, 'SCALE_ILD', None)} scale_ipd={getattr(_DM, 'SCALE_IPD', None)} "
              f"data_module={os.environ.get('DATA_MODULE')}", flush=True)
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

    weight_fn = load_weight_fn(a.loss_weight, device)   # E114
    start_ep = 0; best = 1e9; hist = []
    if a.resume:                                     # continue training: restore model/EMA/opt, fast-forward the schedule
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ck["raw_state"]); ema.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        start_ep = int(ck.get("next_epoch", 0))
        # E144: best/hist ride inside last.pth, so dying before train_done.json no longer resets
        # model selection. train_done.json stays as the fallback for E143-era checkpoints.
        if "best_val_mae_m" in ck:
            best = float(ck["best_val_mae_m"]); hist = list(ck.get("hist", []))
        else:
            _td = os.path.join(os.path.dirname(a.resume), "train_done.json")
            if os.path.exists(_td) and os.path.dirname(a.resume) == rd:
                _j = json.load(open(_td)); best = _j.get("best_val_mae_m", 1e9); hist = _j.get("hist", [])
        if start_ep >= a.epochs:                     # a finished run relaunched by the worker: exit clean
            if is_main:
                print(f"[resume] already trained {start_ep}/{a.epochs} epochs; nothing to do", flush=True)
                if not os.path.exists(os.path.join(rd, "train_done.json")):
                    json.dump({"best_val_mae_m": best, "hist": hist, "args": vars(a)},
                              open(os.path.join(rd, "train_done.json"), "w"), indent=2)
            if ddp:
                dist.destroy_process_group()
            return
        for _ in range(start_ep * steps_per_ep):     # move to the current position on the (extended) cosine
            sched.step()
        if "rng" in ck:                              # dataloader shuffle order continues where it stopped
            set_rng_state(ck["rng"])
        if is_main:
            print(f"[resume] {a.resume} ep{start_ep}-> lr={opt.param_groups[0]['lr']:.2e} "
                  f"best={best:.4f} rng={'restored' if 'rng' in ck else 'MISSING'}", flush=True)
    nbatch = len(tr)
    for ep in range(start_ep, a.epochs):
        if ddp:
            tr_sampler.set_epoch(ep)
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        opt.zero_grad()
        for i, b in enumerate(tr):
            spec = b["spec"][:, :a.nviews * a.in_ch].to(device, non_blocking=True)
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
            d_m = (gt.float() * a.max_depth).clamp(0.05, a.max_depth)   # E114: real-metre range for weighting
            wgt = weight_fn(d_m) * mask
            loss = ((D.float() - gt).abs() * wgt).sum() / wgt.sum().clamp(min=1e-6)   # E114: distance-weighted masked L1
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
            vmae = quick_val(ema, va, device, a.max_depth, wlat, a.nviews * a.in_ch, vp)
            hist.append({"epoch": ep, "loss": run, "val_mae_m": vmae})
            print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s loss={run:.4f} val_MAE={vmae:.4f}m", flush=True)
            if vmae < best:
                best = vmae
                atomic_save({"state_dict": ema.state_dict(), "args": vars(a)}, os.path.join(rd, "best.pth"))
            # E144: the resumable bundle, every epoch (E143 wrote this once, after the last epoch)
            atomic_save(ckpt_bundle(ema, raw_model, opt, sched, ep + 1, best, hist, a),
                        os.path.join(rd, "last.pth"))
        if ddp:
            dist.barrier()
    if is_main:
        # last.pth = resumable bundle: EMA (state_dict, eval-compatible) + raw model + optimizer/scheduler/epoch
        atomic_save(ckpt_bundle(ema, raw_model, opt, sched, a.epochs, best, hist, a),
                    os.path.join(rd, "last.pth"))
        json.dump({"best_val_mae_m": best, "hist": hist, "args": vars(a)},
                  open(os.path.join(rd, "train_done.json"), "w"), indent=2)
        print(f"[done] best val MAE={best:.4f}m -> {rd}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
