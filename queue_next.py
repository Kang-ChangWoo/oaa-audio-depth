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
            [ECO_PY, "-c", "import torch, mmcv, transformers, diffusers"],
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


# (job_factory, ready_predicate) in priority order
JOBS = [(lambda m=m: e40_job(m), lambda m=m: done(f"comparison/oaa_{m}_bmax"))
        for m in ("r2", "fb", "fs", "cb", "r6", "r8")]
JOBS += [(lambda n=n, w=w: eco_job(n, w), eco_ready)
         for n, w in (("eco_r2_wstd", "std"), ("eco_r2_wlong", "long"), ("eco_r2_wnone", "none"))]


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
