"""Persistent GPU-queue dispatcher: run {2,4,6,8-ch} x {all models} on Replica, keep every GPU busy.

Fills all free GPUs, launches the next pending job whenever one frees, skips runs that already have
a best.pth, and never touches a GPU that is already busy (so it coexists with manual runs). Runs
until the whole matrix is done. Launch in background:

  DATA_MODULE=data_0422 R0422_SPLIT=off3 nohup python3 dispatch_replica.py > comparison/logs/dispatch.log 2>&1 &

Stop with: pkill -f dispatch_replica.py  (does not kill the training jobs it started).
"""
import os, sys, time, subprocess

MAIN = os.path.dirname(os.path.abspath(__file__))
PY = "/opt/conda/bin/python"
ECO_PY = "/root/local1/changwoo/echodiff_env/bin/python"
LOGS = os.path.join(MAIN, "comparison", "logs")
os.makedirs(LOGS, exist_ok=True)

MODES = ["r2", "fb", "r6", "r8"]                       # 2 / 4 / 6 / 8 channels
BASE_ENV = {**os.environ, "DATA_MODULE": "data_0422", "R0422_SPLIT": "off3",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}   # reduce fragmentation OOM
ECO_ENV = {**BASE_ENV, "HF_HOME": "/root/local1/changwoo/_echodiff_weights"}
NVIEWS = {"r2": 2, "fb": 4, "r6": 6, "r8": 8}
MEM_FREE_MB = 2000                                     # a GPU is "free" below this
POLL = 30

# per (model, mode) batch size — tuned for a 48GB A6000 ("batch 최대로", 2026-07-23).
# oaa estimates from measured points: fullres+decdeep+msl 4ch bs6 ≈ 21GB(3090), research fullres
# 4ch bs16 ≈ 38GB(A6000) → per-sample ≈ 2.4-2.8GB @4ch, ~linear in nviews. shrink<1 = OOM retry.
def bs(model, mode, shrink=1.0):
    n = NVIEWS[mode]
    if model == "oaa":   b = {2: 24, 4: 14, 6: 9, 8: 7}[n]
    elif model == "beyond": b = 32
    elif model == "eco":   b = 16 if n <= 4 else 12
    elif model in ("vit",): b = 48 if n <= 4 else 32
    else: b = 64 if n <= 4 else 48                     # resnet / echoscan / batvision
    return max(1, int(b * shrink))


def job(model, mode, shrink=1.0):
    """Return (run_name, argv, env) for one model at one channel-mode."""
    stem = {'resnet':'rn','vit':'vit','echoscan':'es','beyond':'byd','batvision':'bat','oaa':'oaa','eco':'eco'}[model]
    # oaa gets a fresh _bmax run name: the old oaa_r2/oaa_fb (bs16/6, 3090) stay as small-batch runs
    run = f"{stem}_{mode}_bmax" if model == "oaa" else f"{stem}_{mode}"
    b = str(bs(model, mode, shrink))
    if model in ("resnet", "vit", "echoscan", "beyond"):
        argv = [PY, "-u", "train_baseline.py", "--model", model, "--run-name", run,
                "--mode", mode, "--batch-size", b, "--num-workers", "8"]
        return run, argv, BASE_ENV
    if model == "batvision":
        argv = [PY, "-u", "train_batvision.py", "--run-name", run, "--mode", mode,
                "--batch-size", b, "--num-workers", "8", "--out-dir", "comparison"]
        return run, argv, BASE_ENV
    if model == "oaa":       # ours: fullres decoder-upgrade + champion recipe + front-relative poses
        argv = [PY, "-u", "train_oaa.py", "--run-name", run, "--nviews", str(NVIEWS[mode]),
                "--data-mode", mode, "--cond-mode", "adaln", "--full-res", "--full-res-enc",
                "--dec-deep", "--multi-scale-lift", "--lr", "5e-4", "--epochs", "40",
                "--batch-size", b, "--num-workers", "8", "--out-dir", "comparison"]
        return run, argv, BASE_ENV
    if model == "eco":       # isolated env (torch 1.13 / SD-UNet / wav2vec2)
        argv = [ECO_PY, "-u", "train_echodiffusion.py", "--run-name", run, "--mode", mode,
                "--batch-size", b, "--num-workers", "8"]
        return run, argv, ECO_ENV
    raise ValueError(model)


# order: ALL oaa modes first (user priority 2026-07-23), then the rest per channel-mode.
# Everything already trained (train_done.json) is skipped, so listing the full matrix is free.
MODEL_ORDER = ["oaa", "resnet", "echoscan", "batvision", "vit", "eco", "beyond"]
QUEUE = [("oaa", mode) for mode in MODES] + \
        [(m, mode) for mode in MODES for m in MODEL_ORDER if m != "oaa"]


def done(run):
    return os.path.exists(os.path.join(MAIN, "comparison", run, "train_done.json"))


def active(run):
    """True if a training process for this run is already alive (avoid duplicate launches).
    NOTE: pattern must not start with '--' (pgrep would parse it as an option and always fail,
    which silently disabled this guard and caused duplicate launches on 2026-07-23)."""
    return subprocess.call(["pgrep", "-f", f"run-name {run}( |$)"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def skip(run):
    return done(run) or active(run)


def free_gpus(busy):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]).decode()
    free = []
    for line in out.strip().splitlines():
        i, used = [x.strip() for x in line.split(",")]
        if int(used) < MEM_FREE_MB and int(i) not in busy:
            free.append(int(i))
    return free


def main():
    pending = list(QUEUE)
    running = {}                                       # gpu -> (proc, run, job_key)
    tries = {}                                         # job_key -> attempts
    print(f"[dispatch] {len(pending)} jobs: {MODES} x {MODEL_ORDER}", flush=True)
    while pending or running:
        # reap finished; requeue transient failures (up to 3 attempts) that produced no train_done
        for g, (p, run, key) in list(running.items()):
            if p.poll() is not None:
                ok = p.returncode == 0 or done(run)
                print(f"[dispatch] GPU{g} done: {run} (exit {p.returncode}){'' if ok else ' [FAILED]'}", flush=True)
                del running[g]
                if not ok and tries.get(key, 1) < 3:
                    pending.append(key)
                    print(f"[dispatch] requeue {run} (attempt {tries.get(key,1)+1})", flush=True)
        # skip already-trained (best.pth present)
        pending = [(m, md) for (m, md) in pending if not skip(job(m, md)[0])]
        # launch onto free gpus
        for g in free_gpus(set(running)):
            if not pending:
                break
            key = pending.pop(0)
            model, mode = key
            shrink = 0.7 ** tries.get(key, 0)          # OOM fallback: retry at 70% batch per attempt
            run, argv, env = job(model, mode, shrink)
            if skip(run):
                continue
            tries[key] = tries.get(key, 0) + 1
            log = open(os.path.join(LOGS, f"{run}.log"), "w")
            e = {**env, "CUDA_VISIBLE_DEVICES": str(g)}
            p = subprocess.Popen(argv, cwd=MAIN, env=e, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            running[g] = (p, run, key)
            print(f"[dispatch] GPU{g} launch: {run}  (bs={bs(model,mode,shrink)}, attempt {tries[key]})", flush=True)
            time.sleep(8)                              # stagger startup
        time.sleep(POLL)
    print("[dispatch] all jobs complete", flush=True)


if __name__ == "__main__":
    main()
