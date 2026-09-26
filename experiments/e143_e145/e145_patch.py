"""E145: build train_oaa_e145.py from train_oaa_e144.py by exact-string replacement.

Every edit is a literal find/replace that must match exactly once, so everything NOT listed here is
byte-identical to train_oaa_e144.py (md5 df08858a74fa39f7c7c9bf4c42427a6f). Scope: validation
metrics + a second checkpoint only. Model, optimizer, schedule, loss, data and RNG consumption are
untouched, so a fresh E145 run follows the same training trajectory as the E144 run with the same
seed -- that is what makes PREREG_E145 gate G1 (reproduce E144) a meaningful test rather than a
tautology.
"""
import hashlib
import os
import sys

SRC = "/root/storage/e144_code/train_oaa_e144.py"
DST = "/root/storage/e145_code/train_oaa_e145.py"
BAK = DST + ".bak-pre-ownobj-20260927"
SRC_MD5 = "df08858a74fa39f7c7c9bf4c42427a6f"

HEADER = '''"""E145: train_oaa_e144.py + per-arm objective-aware model selection. No recipe change.

E144 selected `best.pth` on quick_val's UNWEIGHTED cos-latitude masked L1. For `--loss-weight
invfreq` that is not the quantity the run minimises, so the invfreq arm was scored at a checkpoint
chosen by someone else's objective. E145 records both numbers every epoch and saves both
checkpoints in ONE training run:

  best.pth      selected on val unweighted MAE   -> reproduces E144 (gate G1)
  best_own.pth  selected on val ARM-WEIGHTED MAE -> the arm's own objective

For `--loss-weight none` (the `ctrl` and `bin4` arms) the weight function returns ones, so the two
val metrics are bit-identical (x * 1.0 == x) and best_own.pth MUST equal best.pth. That equality is
E145's integrity check -- see PREREG_E145.md section 4.

Nothing here touches the training step, the optimizer, the schedule or RNG consumption: quick_val
runs under no_grad in eval mode and the added work is arithmetic on tensors it already has. So a
fresh E145 run follows the same trajectory as the E144 run with the same seed.

E144 docstring follows.
"""
'''

QUICK_VAL_OLD = '''@torch.no_grad()
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
'''

QUICK_VAL_NEW = '''@torch.no_grad()
def quick_val(model, va, device, max_depth, wlat, nch, vp=None, weight_fn=None):
    """E145: one val pass, three numbers. Only the first two ever select a checkpoint.

    mae   -- E144's selection metric, byte-for-byte the same formula: cos-latitude masked L1 in
             metres with NO distance weight. Selects best.pth. This is what gate G1 compares.
    obj   -- the SAME formula with this arm's training distance weight multiplied into the
             per-pixel weight, i.e. the arm's own objective measured on val. Selects best_own.pth.
             With --loss-weight none the weight is exactly 1.0 everywhere, so obj is bit-identical
             to mae and best_own.pth comes out equal to best.pth (the integrity check).
    tform -- DIAGNOSTIC ONLY, never selects anything: the training loss formula verbatim
             (normalised depth in [0,1], arm weight, NO cos-latitude term). Logged so the report
             can state whether dropping cos-latitude would have named a different epoch, without
             adding a third checkpoint or a third post-hoc selection rule.
    """
    model.eval(); tot = wn = 0.0; otot = own = 0.0; ttot = twn = 0.0
    for b in va:
        sp = b["spec"][:, :nch].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Dn = model(sp, view_poses=vp).float()
        D = Dn * max_depth
        gtn = b["depth"].to(device)
        gt = gtn * max_depth
        m = b["mask"].to(device)
        w = wlat * m
        tot += ((D - gt).abs() * w).sum().item(); wn += w.sum().item()
        # same clamp as the training loss: d_m = (gt * max_depth).clamp(0.05, max_depth)
        dw = weight_fn(gt.clamp(0.05, max_depth)) if weight_fn is not None else torch.ones_like(gt)
        wo = w * dw
        otot += ((D - gt).abs() * wo).sum().item(); own += wo.sum().item()
        wt = m * dw
        ttot += ((Dn - gtn).abs() * wt).sum().item(); twn += wt.sum().item()
    return tot / max(wn, 1e-6), otot / max(own, 1e-6), ttot / max(twn, 1e-6)
'''

EDITS = [
    # 1. ckpt_bundle carries the own-objective best so a resume cannot reset it
    ('''def ckpt_bundle(ema, raw_model, opt, sched, next_epoch, best, hist, a):
    return {"state_dict": ema.state_dict(), "raw_state": raw_model.state_dict(),
            "opt": opt.state_dict(), "sched_last_epoch": sched.last_epoch,
            "next_epoch": next_epoch, "best_val_mae_m": best, "hist": hist,
            "rng": rng_state(), "args": vars(a)}
''',
     '''def ckpt_bundle(ema, raw_model, opt, sched, next_epoch, best, hist, a, best_own=1e9):
    return {"state_dict": ema.state_dict(), "raw_state": raw_model.state_dict(),
            "opt": opt.state_dict(), "sched_last_epoch": sched.last_epoch,
            "next_epoch": next_epoch, "best_val_mae_m": best, "hist": hist,
            "best_val_obj_m": best_own,          # E145: own-objective selection survives a resume
            "rng": rng_state(), "args": vars(a)}
'''),

    # 2. quick_val returns all three val numbers from a single pass
    (QUICK_VAL_OLD, QUICK_VAL_NEW),

    # 3. initialise the second best tracker
    ('''    weight_fn = load_weight_fn(a.loss_weight, device)   # E114
    start_ep = 0; best = 1e9; hist = []
''',
     '''    weight_fn = load_weight_fn(a.loss_weight, device)   # E114
    start_ep = 0; best = 1e9; hist = []
    best_own = 1e9                                   # E145: best val ARM-WEIGHTED MAE so far
'''),

    # 4. resume also restores the own-objective best
    ('''        if "best_val_mae_m" in ck:
            best = float(ck["best_val_mae_m"]); hist = list(ck.get("hist", []))
''',
     '''        if "best_val_mae_m" in ck:
            best = float(ck["best_val_mae_m"]); hist = list(ck.get("hist", []))
            best_own = float(ck.get("best_val_obj_m", 1e9))   # E145
'''),

    # 5. the early-clean-exit path records both bests
    ('''                if not os.path.exists(os.path.join(rd, "train_done.json")):
                    json.dump({"best_val_mae_m": best, "hist": hist, "args": vars(a)},
                              open(os.path.join(rd, "train_done.json"), "w"), indent=2)
''',
     '''                if not os.path.exists(os.path.join(rd, "train_done.json")):
                    json.dump({"best_val_mae_m": best, "best_val_obj_m": best_own,
                               "hist": hist, "args": vars(a)},
                              open(os.path.join(rd, "train_done.json"), "w"), indent=2)
'''),

    # 6. the epoch loop: log three numbers, save two checkpoints
    ('''            vmae = quick_val(ema, va, device, a.max_depth, wlat, a.nviews * a.in_ch, vp)
            hist.append({"epoch": ep, "loss": run, "val_mae_m": vmae})
            print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s loss={run:.4f} val_MAE={vmae:.4f}m", flush=True)
            if vmae < best:
                best = vmae
                atomic_save({"state_dict": ema.state_dict(), "args": vars(a)}, os.path.join(rd, "best.pth"))
            # E144: the resumable bundle, every epoch (E143 wrote this once, after the last epoch)
            atomic_save(ckpt_bundle(ema, raw_model, opt, sched, ep + 1, best, hist, a),
                        os.path.join(rd, "last.pth"))
''',
     '''            vmae, vobj, vtform = quick_val(ema, va, device, a.max_depth, wlat,
                                           a.nviews * a.in_ch, vp, weight_fn)
            hist.append({"epoch": ep, "loss": run, "val_mae_m": vmae,
                         "val_obj_m": vobj, "val_trainform": vtform})
            print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s loss={run:.4f} val_MAE={vmae:.4f}m "
                  f"val_OBJ={vobj:.4f}m val_TFORM={vtform:.5f}", flush=True)
            if vmae < best:
                best = vmae
                atomic_save({"state_dict": ema.state_dict(), "args": vars(a)}, os.path.join(rd, "best.pth"))
            if vobj < best_own:        # E145: selection on THIS arm's own objective
                best_own = vobj
                atomic_save({"state_dict": ema.state_dict(), "args": vars(a)},
                            os.path.join(rd, "best_own.pth"))
            # E144: the resumable bundle, every epoch (E143 wrote this once, after the last epoch)
            atomic_save(ckpt_bundle(ema, raw_model, opt, sched, ep + 1, best, hist, a, best_own),
                        os.path.join(rd, "last.pth"))
'''),

    # 7. final save + train_done.json record both bests
    ('''        atomic_save(ckpt_bundle(ema, raw_model, opt, sched, a.epochs, best, hist, a),
                    os.path.join(rd, "last.pth"))
        json.dump({"best_val_mae_m": best, "hist": hist, "args": vars(a)},
                  open(os.path.join(rd, "train_done.json"), "w"), indent=2)
        print(f"[done] best val MAE={best:.4f}m -> {rd}", flush=True)
''',
     '''        atomic_save(ckpt_bundle(ema, raw_model, opt, sched, a.epochs, best, hist, a, best_own),
                    os.path.join(rd, "last.pth"))
        _bep = min(hist, key=lambda h: h["val_mae_m"])["epoch"] if hist else -1
        _oep = min(hist, key=lambda h: h["val_obj_m"])["epoch"] if hist else -1
        _tep = min(hist, key=lambda h: h["val_trainform"])["epoch"] if hist else -1
        json.dump({"best_val_mae_m": best, "best_val_obj_m": best_own,
                   "best_ep_mae": _bep, "best_ep_obj": _oep, "best_ep_trainform": _tep,
                   "hist": hist, "args": vars(a)},
                  open(os.path.join(rd, "train_done.json"), "w"), indent=2)
        print(f"[done] best val MAE={best:.4f}m @ep{_bep} | best val OBJ={best_own:.4f}m @ep{_oep} "
              f"| trainform @ep{_tep} -> {rd}", flush=True)
'''),
]


def main():
    src = open(SRC, "rb").read()
    got = hashlib.md5(src).hexdigest()
    if got != SRC_MD5:
        sys.exit(f"REFUSE: {SRC} md5 {got} != expected {SRC_MD5}")
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    if not os.path.exists(BAK):
        open(BAK, "wb").write(src)
    txt = HEADER + src.decode()
    for i, (old, new) in enumerate(EDITS, 1):
        n = txt.count(old)
        if n != 1:
            sys.exit(f"REFUSE: edit {i} matched {n} times, expected 1:\n{old[:160]}")
        txt = txt.replace(old, new)
    open(DST, "w").write(txt)
    print("wrote", DST, "md5", hashlib.md5(txt.encode()).hexdigest())
    print("pristine E144 backup:", BAK, "md5", hashlib.md5(src).hexdigest())
    print(f"{len(EDITS)} edits, each matched exactly once")


if __name__ == "__main__":
    main()
