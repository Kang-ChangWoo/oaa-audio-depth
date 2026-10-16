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
import os, json, math, time, argparse, copy, random
import torch.distributed as dist
import numpy as np
import torch

from core.data import get_data_module
from core.metrics import cos_lat, depth_loss
from model.oaa import (OAAv2Depth, VIEW_FUSIONS, RAY_GATE_MODES, GEO_BASES,
                       IDENTITY_INIT_PARAM_SUFFIXES, yaw_roll_target)

_DM = get_data_module()          # DATA_MODULE=data_mp3d (Matterport3D, default) | data_0422 (Replica)
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
    # attention / fusion / loss / masking variants (all default-off => reference recipe)
    p.add_argument("--qk-norm", action="store_true")       # E1: RMS-normalised ray-cross Q/K + learnable head scales
    p.add_argument("--ray-gate", action="store_true")      # E2: identity-initialised per-(ray,head) output gate
    p.add_argument("--view-fusion", default="global", choices=list(VIEW_FUSIONS))   # E4
    p.add_argument("--depth-loss", default="plain", choices=["plain", "spherical"])  # E5
    p.add_argument("--explicit-obs-mask", action="store_true")   # E6: true masking instead of zeroed spectrograms
    p.add_argument("--dump-attn-stats", action="store_true")     # diagnostics only, never feeds the loss
    # parameter-level tuning of the variants above. Every default reproduces the reference exactly.
    # Motivated by one measurement: the zero/identity-initialised parameters barely move in 40
    # epochs (logit_scale 3-11 %, fine_to_dec RMS 0.0057 vs the decoder's own 0.0173).
    p.add_argument("--param-lr-mult", type=float, default=1.0)   # LR multiplier for exactly that set
    p.add_argument("--logit-scale-init", type=float, default=0.0)   # log-scale; 0 -> multiplier 1
    p.add_argument("--geo-scale-init", type=float, default=0.0)
    p.add_argument("--ray-gate-mode", default="scalar", choices=list(RAY_GATE_MODES))
    p.add_argument("--ray-gate-init", type=float, default=0.0)   # gate = 2*sigmoid(init); 0 -> 1.0
    p.add_argument("--fine-skip-init", type=float, default=0.0)  # std of the fine-skip conv init
    # architecture repairs (archfix_, 2026-10-16). All default off => reference recipe, bit for bit.
    p.add_argument("--fine-layerscale", type=float, default=0.0)   # B1: LayerScale gamma on the fine skip (0 = off)
    p.add_argument("--registers", type=int, default=0)             # B2: register tokens in the ray cross-attention
    p.add_argument("--scene-registers", action="store_true")       # B3: registers carry a pooled scene code
    p.add_argument("--geo-basis", default="raw", choices=list(GEO_BASES))   # B4: raw 5 numbers | degree-4 real SH
    # far-field / RMSE-directed switches (2026-10-14). All default off => reference recipe, bit for bit.
    p.add_argument("--aux-l2", type=float, default=0.0)      # additive masked MSE weight; L1 stays the objective
    p.add_argument("--erp-circular", action="store_true")    # decoder convs wrap the ERP azimuth seam (no new params)
    p.add_argument("--yaw-roll-aug", action="store_true")    # per batch: roll the ERP target, shift declared yaws (exact symmetry)
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

    if a.explicit_obs_mask and not a.subset_aug:
        print("[warn] --explicit-obs-mask without --subset-aug: every observation is always valid, "
              "so the mask is all-ones and training is identical to the reference recipe", flush=True)
    model = OAAv2Depth(C=a.dim, nviews=a.nviews, rounds=a.rounds, lh=a.lift_h, lw=a.lift_w,
                       stem_stride1=a.stem_stride1, max_depth=a.max_depth,
                       no_pose_emb=a.no_pose_emb, no_ray_emb=a.no_ray_emb, no_geo_bias=a.no_geo_bias,
                       no_tf_pe=a.no_tf_pe, no_cross=a.no_cross,
                       qk_norm=a.qk_norm, ray_gate=a.ray_gate, view_fusion=a.view_fusion,
                       logit_scale_init=a.logit_scale_init, geo_scale_init=a.geo_scale_init,
                       ray_gate_mode=a.ray_gate_mode, ray_gate_init=a.ray_gate_init,
                       fine_skip_init=a.fine_skip_init, erp_circular=a.erp_circular,
                       fine_layerscale=a.fine_layerscale, registers=a.registers,
                       scene_registers=a.scene_registers, geo_basis=a.geo_basis).to(device)
    if is_main:
        print(f"[cfg] {vars(a)} params={sum(x.numel() for x in model.parameters())/1e6:.2f}M ddp_world={world}", flush=True)
        # run git in the REPO, not in the launcher's cwd: `git_dirty` / `git_diff` were null in every
        # manifest written so far, because a launcher that cd'd anywhere else got "not a repository"
        # on stderr and an empty stdout, which `or None` then recorded as "clean". git_sha survived
        # only through the _sha_nogit fallback below, which reads .git/HEAD by path.
        _REPO = os.path.dirname(os.path.abspath(__file__))
        _sh = lambda c: (__import__("subprocess").run(c, shell=True, capture_output=True, text=True,
                                                      cwd=_REPO).stdout.strip() or None)
        # Provenance the run directory could not answer before (2026-10-14). The STFT settings are
        # module-level env reads in the data module (STFT_NFFT / STFT_WIN / STFT_HOP, plus MP3D's
        # STFT_WINDOW crop), so a manifest without them cannot tell "ran at the released hop 160"
        # apart from "ran with STFT_HOP exported to something else" -- that had to be re-derived by
        # reading the launcher and the live process environment. Stamp what the LOADED data module
        # actually holds, not the env, so an unset env is recorded as the default that was used;
        # env_overrides separately records which of them were explicitly exported.
        _stft = {k: getattr(_DM, k, None) for k in ("N_FFT", "WIN", "HOP", "WINDOW")}
        _stft = {"STFT_NFFT": _stft["N_FFT"], "STFT_WIN": _stft["WIN"], "STFT_HOP": _stft["HOP"],
                 "STFT_WINDOW": _stft["WINDOW"],
                 "env_overrides": {k: os.environ[k] for k in
                                   ("STFT_NFFT", "STFT_WIN", "STFT_HOP", "STFT_WINDOW")
                                   if k in os.environ}}
        def _sha_nogit():   # the training container has no git binary -> every manifest so far wrote null
            try:
                g = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".git")
                h = open(os.path.join(g, "HEAD")).read().strip()
                if not h.startswith("ref: "):
                    return h
                ref = h[5:]
                if os.path.exists(os.path.join(g, ref)):
                    return open(os.path.join(g, ref)).read().strip()
                for ln in open(os.path.join(g, "packed-refs")):
                    if ln.rstrip().endswith(" " + ref):
                        return ln.split()[0]
            except Exception:
                return None
            return None
        json.dump({"args": vars(a), "params_m": sum(x.numel() for x in model.parameters()) / 1e6,
                   "git_sha": _sh("git rev-parse HEAD") or _sha_nogit(), "git_dirty": _sh("git status --short"),
                   "git_diff": _sh("git diff"), "data_module": a.data_module, "dmode": dmode,
                   "stft": _stft,
                   "view_poses": vp, "split_sizes": {"train": len(tr.dataset), "val": len(va.dataset)},
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
                   "steps_per_ep_note": "filled after the scheduler is built"},
                  open(os.path.join(rd, "manifest.json"), "w"), indent=2, default=str)
    raw_model = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])],
                                                          find_unused_parameters=True)  # aux_head is not in the loss

    if a.param_lr_mult == 1.0:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)   # reference path
    else:
        # one parameter group for the zero/identity-initialised set, one for everything else. The
        # LambdaLR below scales both groups by the same factor, so the ratio is held all the way
        # through warmup and cosine decay.
        idm, rest = [], []
        for n, q in raw_model.named_parameters():
            (idm if n.endswith(IDENTITY_INIT_PARAM_SUFFIXES) else rest).append(q)
        opt = torch.optim.AdamW([{"params": rest, "lr": a.lr},
                                 {"params": idm, "lr": a.lr * a.param_lr_mult}], weight_decay=a.wd)
        if is_main:
            print(f"[param-lr] {len(idm)} identity-init tensors at lr x{a.param_lr_mult}, "
                  f"{len(rest)} others at lr {a.lr}", flush=True)
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
            vp_in, spec_in, ov_in = vp, spec, None
            _vp = a.vdrop_p * min(1.0, max(0.0, (ep - a.vdrop_start + 1) / max(1, a.vdrop_ramp)))
            if a.subset_aug and random.random() < _vp:          # observation-token masking (curriculum ramp)
                # never zero every observation: k < nviews
                _pool = [x for x in range(min(a.vdrop_kstep, 2), a.vdrop_kmax + 1, a.vdrop_kstep)
                         if x < spec_in.shape[1]] or [1]
                k = random.choice(_pool)
                idx = random.sample(range(spec_in.shape[1]), k)
                spec_in = spec_in.clone(); spec_in[:, idx] = 0
                if a.explicit_obs_mask:
                    # (B,N) even though this dropout is per-batch, so the model API is already
                    # per-sample for a future per-sample curriculum. view_poses stay untouched:
                    # removing entries would break the fixed view indexing and the ckpt contract.
                    ov_in = torch.ones(spec_in.shape[0], a.nviews, dtype=torch.bool, device=device)
                    ov_in[:, idx] = False
            if a.yaw_roll_aug:
                # same recording, receiver declared k columns further round, target rolled to match.
                # k is a whole number of fine-grid columns (2*lw of them across 512), so the fine-lift
                # roll in model.oaa._fine_lift is exact; the coarse geometry is continuous in yaw.
                # The RNG is only drawn when the flag is on, so the default recipe's stream is untouched.
                _step = gt.shape[-1] // (2 * raw_model.lw)
                k = _step * random.randrange(2 * raw_model.lw)
                gt, mask, vp_in = yaw_roll_target(gt, mask, vp_in if vp_in is not None else raw_model.view_pose,
                                                  k, gt.shape[-1])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                D = model(spec_in, view_poses=vp_in, obs_valid=ov_in)
            loss = depth_loss(D, gt, mask, a.depth_loss, wlat, a.aux_l2)
            if loss is None:                                           # no weighted-valid pixel anywhere
                # drop this micro-batch's gradient but keep the step cadence, so --accum still
                # defines the effective batch and the LR schedule stays on its pre-registered horizon
                print(f"[skip] ep{ep} batch{i}: empty depth mask, micro-batch dropped", flush=True)
            else:
                (loss / accum).backward()                              # accumulate grads over `accum` micro-batches
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
            if a.dump_attn_stats and not a.no_cross:
                ema.ray_mic.collect_stats = True        # sampled during val only; cannot touch training
            vmae = quick_val(ema, va, device, a.max_depth, wlat, a.nviews, vp)
            hist.append({"epoch": ep, "loss": run, "val_mae_m": vmae})
            print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s loss={run:.4f} val_MAE={vmae:.4f}m", flush=True)
            if a.dump_attn_stats and not a.no_cross:
                ema.ray_mic.collect_stats = False
                _st = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in ema.ray_mic.stats.items()}
                with open(os.path.join(rd, "attn_stats.jsonl"), "a") as _f:
                    _f.write(json.dumps({"epoch": ep, **_st}) + "\n")
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
