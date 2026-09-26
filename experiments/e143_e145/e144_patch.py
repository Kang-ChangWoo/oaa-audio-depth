"""E144: build train_oaa_e144.py from train_oaa_e143.py by exact-string replacement.

Every edit is a literal find/replace that must match exactly once, so everything NOT listed here
is byte-identical to train_oaa_e143.py (md5 2926c74210175dc03d9dc5591bb76566). Scope: crash-safe
checkpointing only -- no change to model, optimizer, schedule, loss or data.
"""
import hashlib
import os
import sys

SRC = "/root/storage/e143_code/train_oaa_e143.py"
DST = "/root/storage/e144_code/train_oaa_e144.py"
BAK = DST + ".bak-pre-perepoch-ckpt-20260926"
SRC_MD5 = "2926c74210175dc03d9dc5591bb76566"

HEADER = '''"""E144: train_oaa_e143.py + crash-safe checkpointing. No recipe change.

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
'''

HELPERS = '''

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

'''

EDITS = [
    # 1. helpers, right after the cue table
    ('CUE_CH = {"none": 1, "ild": 2, "ipd": 3, "bin4": 4}   # channels per ear observation\n',
     'CUE_CH = {"none": 1, "ild": 2, "ipd": 3, "bin4": 4}   # channels per ear observation\n' + HELPERS),

    # 2. --resume auto resolves against this run's own dir
    ('''    rd = os.path.join(a.out_dir, a.run_name)
    if is_main:
        os.makedirs(rd, exist_ok=True)
''',
     '''    rd = os.path.join(a.out_dir, a.run_name)
    if is_main:
        os.makedirs(rd, exist_ok=True)
    if a.resume == "auto":       # E144: relaunch-safe -- resume iff this run dir already has a last.pth
        _auto = os.path.join(rd, "last.pth")
        a.resume = _auto if os.path.exists(_auto) else ""
        print(f"[resume] auto -> {a.resume or 'none (fresh start)'}", flush=True)
'''),

    # 3. resume block: best/hist/RNG from the bundle, graceful no-op when already finished
    ('''    start_ep = 0
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

    weight_fn = load_weight_fn(a.loss_weight, device)   # E114
    best = 1e9; hist = []
    if a.resume:                                     # resuming in the same run dir: keep the existing best
        _td = os.path.join(os.path.dirname(a.resume), "train_done.json")
        if os.path.exists(_td) and os.path.dirname(a.resume) == rd:
            _j = json.load(open(_td)); best = _j.get("best_val_mae_m", 1e9); hist = _j.get("hist", [])
''',
     '''    weight_fn = load_weight_fn(a.loss_weight, device)   # E114
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
'''),

    # 4. per-epoch atomic last.pth (and atomic best.pth)
    ('''            if vmae < best:
                best = vmae
                torch.save({"state_dict": ema.state_dict(), "args": vars(a)}, os.path.join(rd, "best.pth"))
        if ddp:
            dist.barrier()
''',
     '''            if vmae < best:
                best = vmae
                atomic_save({"state_dict": ema.state_dict(), "args": vars(a)}, os.path.join(rd, "best.pth"))
            # E144: the resumable bundle, every epoch (E143 wrote this once, after the last epoch)
            atomic_save(ckpt_bundle(ema, raw_model, opt, sched, ep + 1, best, hist, a),
                        os.path.join(rd, "last.pth"))
        if ddp:
            dist.barrier()
'''),

    # 5. final save goes through the same atomic bundle
    ('''        torch.save({"state_dict": ema.state_dict(), "raw_state": raw_model.state_dict(),
                    "opt": opt.state_dict(), "sched_last_epoch": sched.last_epoch,
                    "next_epoch": a.epochs, "args": vars(a)}, os.path.join(rd, "last.pth"))
''',
     '''        atomic_save(ckpt_bundle(ema, raw_model, opt, sched, a.epochs, best, hist, a),
                    os.path.join(rd, "last.pth"))
'''),
]


def main():
    src = open(SRC, "rb").read()
    got = hashlib.md5(src).hexdigest()
    if got != SRC_MD5:
        sys.exit(f"REFUSE: {SRC} md5 {got} != expected {SRC_MD5}")
    if not os.path.exists(BAK):
        open(BAK, "wb").write(src)
    txt = HEADER + src.decode()
    for old, new in EDITS:
        n = txt.count(old)
        if n != 1:
            sys.exit(f"REFUSE: edit matched {n} times, expected 1:\n{old[:120]}")
        txt = txt.replace(old, new)
    open(DST, "w").write(txt)
    print("wrote", DST, "md5", hashlib.md5(txt.encode()).hexdigest())
    print("pristine backup:", BAK, "md5", hashlib.md5(src).hexdigest())


if __name__ == "__main__":
    main()
