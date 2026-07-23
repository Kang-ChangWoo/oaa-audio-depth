"""Unified follow-up queue (replaces relaunch_e40.py — single owner of free GPUs, no races).

Jobs, in priority order:
  1. Six 40-epoch twins of the running 30-epoch oaa_*_bmax Replica runs — each gated on its
     30ep run finishing (frees that GPU; cosine/EMA make in-place extension impossible).
  2. Three EchoDiffusion waveform-branch ablations on the MP3D setup (user 2026-07-23):
     eco_r2_wstd (CIDE, 10m-cut wave) / eco_r2_wlong (CIDE, 1.0s wave) / eco_r2_wnone (no CIDE).
     Gated on the rebuilt isolated env importing cleanly.

  nohup /opt/conda/bin/python queue_next.py > comparison/logs/queue_next.log 2>&1 &
"""
import os, subprocess, time

MAIN = os.path.dirname(os.path.abspath(__file__))
PY = "/opt/conda/bin/python"
ECO_PY = "/root/local1/changwoo/echodiff_env/bin/python"
BASE = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
REP_ENV = {**BASE, "DATA_MODULE": "data_0422", "R0422_SPLIT": "off3"}
ECO_ENV = {**BASE, "DATA_MODULE": "data_mp3d", "HF_HOME": "/root/local1/changwoo/_echodiff_weights"}
ECO_ENV.pop("PYTORCH_CUDA_ALLOC_CONF", None)   # torch 1.13 doesn't know expandable_segments -> hard crash

BS = {"r2": 24, "fb": 14, "fs": 14, "cb": 14, "r6": 9, "r8": 7}
NV = {"r2": 2, "fb": 4, "fs": 4, "cb": 4, "r6": 6, "r8": 8}


def done(rel):
    return os.path.exists(f"{MAIN}/{rel}/train_done.json")


_eco_ok = False
def eco_ready():
    global _eco_ok
    if not _eco_ok:                                    # cache success only; retry while failing
        _eco_ok = os.path.exists(ECO_PY) and subprocess.call(
            [ECO_PY, "-c", "import torch, mmcv, transformers"],   # diffusers unused by the model
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=MAIN) == 0
    return _eco_ok


def already_running(run):
    """pattern must not start with '--' (pgrep option parsing) — see dispatch_replica.active()."""
    return subprocess.call(["pgrep", "-f", f"run-name {run}( |$)"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def e40_job(m):
    argv = [PY, "-u", "train_oaa.py", "--run-name", f"oaa_{m}_e40", "--nviews", str(NV[m]),
            "--data-mode", m, "--cond-mode", "adaln", "--full-res", "--full-res-enc",
            "--dec-deep", "--multi-scale-lift", "--lr", "5e-4", "--epochs", "40",
            "--batch-size", str(BS[m]), "--num-workers", "8", "--out-dir", "comparison"]
    return f"oaa_{m}_e40", argv, REP_ENV, "comparison"


def eco_job(name, wave_mode):
    argv = [ECO_PY, "-u", "train_echodiffusion.py", "--run-name", name, "--mode", "r2",
            "--wave-mode", wave_mode, "--epochs", "40", "--batch-size", "16",
            "--num-workers", "8", "--out-dir", "comparison_mp3d"]
    return name, argv, ECO_ENV, "comparison_mp3d"


# ---- nearest-STFT baseline retraining round (user 2026-07-23: 선행 연구들도 nearest로) ----
# All prior Replica baselines were trained on the old machine with bilinear STFT; retrain the
# whole matrix at 40 epochs / 48GB batches under the corrected pipeline. New run names *_n keep
# the bilinear round intact for reference.
ECO_REP_ENV = {**ECO_ENV, "DATA_MODULE": "data_0422", "R0422_SPLIT": "off3"}
BASE_BS = {"batvision": {"r2": 64, "fb": 64, "r6": 48, "r8": 48},
           "resnet":    {"r2": 64, "fb": 64, "r6": 48, "r8": 48},
           "echoscan":  {"r2": 64, "fb": 64, "r6": 48, "r8": 48},
           "vit":       {"r2": 48, "fb": 48, "r6": 32, "r8": 32},
           "beyond":    {"r2": 32, "fb": 32, "r6": 32, "r8": 32},
           "eco":       {"r2": 16, "fb": 16, "r6": 12, "r8": 12}}
_STEM = {"resnet": "rn", "vit": "vit", "echoscan": "es", "beyond": "byd",
         "batvision": "bat", "eco": "eco"}


def baseline_job(model, mode):
    run = f"{_STEM[model]}_{mode}_fin"
    b = str(BASE_BS[model][mode])
    if model == "batvision":
        argv = [PY, "-u", "train_batvision.py", "--run-name", run, "--mode", mode, "--epochs", "40",
                "--batch-size", b, "--num-workers", "8", "--out-dir", "comparison"]
        return run, argv, REP_ENV, "comparison"
    if model == "eco":
        argv = [ECO_PY, "-u", "train_echodiffusion.py", "--run-name", run, "--mode", mode,
                "--wave-mode", "std", "--epochs", "40", "--batch-size", b,
                "--num-workers", "8", "--out-dir", "comparison"]
        return run, argv, ECO_REP_ENV, "comparison"
    argv = [PY, "-u", "train_baseline.py", "--model", model, "--run-name", run, "--mode", mode,
            "--epochs", "40", "--batch-size", b, "--num-workers", "8", "--out-dir", "comparison"]
    return run, argv, REP_ENV, "comparison"


# (job_factory, ready_predicate) in priority order
JOBS = [(lambda n=n, w=w: eco_job(n, w), eco_ready)
        for n, w in (("eco_r2_wstd", "std"), ("eco_r2_wlong", "long"), ("eco_r2_wnone", "none"))]
JOBS += [(lambda m=m: e40_job(m), lambda m=m: done(f"comparison/oaa_{m}_bmax"))
         for m in ("r2", "fb", "r6")]   # r8 replaced by the accum-2 fix (oaa_r8_e40a2) below
JOBS += [(lambda m=m: e40_job(m), lambda: True) for m in ("fs", "cb")]   # bmax killed early -> run e40 directly
JOBS += [(lambda mm=(model, mode): baseline_job(*mm), (eco_ready if model == "eco" else (lambda: True)))
         for model in ("batvision", "eco", "echoscan", "vit", "resnet", "beyond")
         for mode in ("r2",)]   # ch2 first (user); fb/r6/r8 rounds follow after review


def r8a2_job():
    # 8ch batch-parity fix (2026-07-23): bmax/e40 ran r8 at bs7 (fullres 8ch = 46.8GB memory
    # wall) vs 14 at 4ch — the gradient-noise mismatch is the prime suspect for r8's val/test
    # regression (0.2981/0.2640 vs r6 0.2707/0.2416). accum 2 -> effective batch 14, same memory.
    argv = [PY, "-u", "train_oaa.py", "--run-name", "oaa_r8_e40a2", "--nviews", "8",
            "--data-mode", "r8", "--cond-mode", "adaln", "--full-res", "--full-res-enc",
            "--dec-deep", "--multi-scale-lift", "--lr", "5e-4", "--epochs", "40",
            "--batch-size", "7", "--accum", "2", "--num-workers", "8", "--out-dir", "comparison"]
    return "oaa_r8_e40a2", argv, REP_ENV, "comparison"


JOBS += [(r8a2_job, lambda: True)]


def free_gpus(busy):
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                   "--format=csv,noheader,nounits"]).decode()
    return [int(i) for i, u in (l.split(",") for l in out.strip().splitlines())
            if int(u) < 2000 and int(i) not in busy]


pending = list(JOBS)
running = {}                                           # gpu -> (proc, run)
while pending or running:
    for g, (p, run) in list(running.items()):
        if p.poll() is not None:
            print(f"[qn] GPU{g} finished {run} (rc={p.returncode})", flush=True)
            del running[g]
    for g in free_gpus(set(running)):
        ready = next((j for j in pending if j[1]()), None)
        if ready is None:
            break
        pending.remove(ready)
        run, argv, env, outdir = ready[0]()
        if done(f"{outdir}/{run}") or already_running(run):
            continue
        log = open(f"{MAIN}/comparison/logs/{run}.log", "w")
        p = subprocess.Popen(argv, cwd=MAIN, env={**env, "CUDA_VISIBLE_DEVICES": str(g)},
                             stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        running[g] = (p, run)
        print(f"[qn] GPU{g} launch {run}", flush=True)
        time.sleep(8)
    time.sleep(60)
print("[qn] all follow-up jobs done", flush=True)
