"""Watch the six running 30-epoch oaa_*_bmax runs; as each finishes, launch its 40-epoch twin
(oaa_{mode}_e40, same fullres+dec-deep+msl recipe and max batch) on a free GPU.

Background: the user's epoch-40 directive was encoded in dispatch_mp3d.py but dispatch_replica.py
never passed --epochs, so the bmax round ran at the 30 default. This watcher upgrades the round
to 40 without killing anything mid-run.

  nohup /opt/conda/bin/python relaunch_e40.py > comparison/logs/e40_watcher.log 2>&1 &
"""
import os, subprocess, time

MAIN = os.path.dirname(os.path.abspath(__file__))
PY = "/opt/conda/bin/python"
ENV = {**os.environ, "DATA_MODULE": "data_0422", "R0422_SPLIT": "off3",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
BS = {"r2": 24, "fb": 14, "fs": 14, "cb": 14, "r6": 9, "r8": 7}
NV = {"r2": 2, "fb": 4, "fs": 4, "cb": 4, "r6": 6, "r8": 8}
PENDING = {m: f"oaa_{m}_bmax" for m in ("r2", "fb", "fs", "cb", "r6", "r8")}  # mode -> 30ep run to wait on


def done(run):
    return os.path.exists(f"{MAIN}/comparison/{run}/train_done.json")


def free_gpu(exclude):
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                   "--format=csv,noheader,nounits"]).decode()
    for line in out.strip().splitlines():
        i, used = [x.strip() for x in line.split(",")]
        if int(used) < 2000 and int(i) not in exclude:
            return int(i)
    return None


launched = {}                                          # mode -> gpu (only to avoid double-assigning)
while PENDING:
    for m, run30 in list(PENDING.items()):
        run40 = f"oaa_{m}_e40"
        if done(run40):
            del PENDING[m]; continue
        if not done(run30):
            continue
        g = free_gpu(set(launched.values()))
        if g is None:
            continue
        argv = [PY, "-u", "train_oaa.py", "--run-name", run40, "--nviews", str(NV[m]),
                "--data-mode", m, "--cond-mode", "adaln", "--full-res", "--full-res-enc",
                "--dec-deep", "--multi-scale-lift", "--lr", "5e-4", "--epochs", "40",
                "--batch-size", str(BS[m]), "--num-workers", "8", "--out-dir", "comparison"]
        log = open(f"{MAIN}/comparison/logs/{run40}.log", "w")
        subprocess.Popen(argv, cwd=MAIN, env={**ENV, "CUDA_VISIBLE_DEVICES": str(g)},
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"[e40] GPU{g} launch {run40} (bs={BS[m]})", flush=True)
        launched[m] = g
        del PENDING[m]
        time.sleep(8)
    time.sleep(60)
print("[e40] all 40-epoch twins launched", flush=True)
