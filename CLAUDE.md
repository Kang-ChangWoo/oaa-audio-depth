# CLAUDE.md — oaa-audio-depth (OAA / HEAR360)

Shared research discipline (number provenance, verdict discipline, PyTorch correctness,
watchers, where results go) lives in the **`research-coding-rules` skill**. Follow it; it is
deliberately not repeated here.

Repo documents to read first:

- **`README.md`** — model, input recipe, channel modes, train/eval commands, checkpoint layout.
- **`SCRIPTS.md`** — per-script index: purpose, what it reads/writes, which table or figure it
  backs, the full environment-variable list, and the checkpoint / run-dir contract.

Everything below is specific to this repository and was verified in these files.

## 1. This repository is PUBLIC, and it has two faces

`github.com/Kang-ChangWoo/oaa-audio-depth` is public. The README is written as an **anonymised
paper release**: no machine paths, no node names, `env.local.sh` deliberately untracked. Keep it
that way — do not commit absolute lab paths, node hostnames or internal run names into tracked
files, and treat any push as publication.

Two code families live side by side:

- **Release side** — `core/`, `model/`, `data_0422.py`, `data_mp3d.py`, `train_oaa.py`,
  `train_batvision.py`, `train_baseline.py`, `train_echodiffusion.py`, `eval*.py`,
  `analysis/`, `viz/`, `tools/`, `configs/`, `comparison/`, `comparison_mp3d/`. Documented in
  README + SCRIPTS.md.
- **Live research side** — the `0820_*` campaign scripts and queues, `data_micgain.py`,
  `experiments/<name>/`. **Not in SCRIPTS.md**, so its conventions are below.

## 2. The real commands

`source env.local.sh` first — it is untracked and holds this machine's paths.

```bash
# Replica, 8 observations: the reported recipe
DATA_MODULE=data_0422 python train_oaa.py --run-name oaa_r8 --nviews 8 --lr 5e-4 --warmup-ep 4 \
    --epochs 40 --batch-size 3 --accum 11 --subset-aug --vdrop-kmax 4 --out-dir comparison
DATA_MODULE=data_0422 python eval.py --run-name oaa_r8 ...   # -> comparison/compare.json
```

`configs/{replica,mp3d}/<run>.json` holds the exact `train_done.json["args"]` of every run
behind a reported number. Reproduce from there, key by key, not from the README prose.

## 3. Data paths on this node (from `env.local.sh`)

| variable | path | note |
|---|---|---|
| `REPLICA_ROOT` | `/root/local1/changwoo/replica_0422_lite` | node-local disk |
| `MP3D_ROOT` | `/root/local1/changwoo/matterport3d_0303renew` | node-local disk |
| `MICGAIN_ROOT` | default `/root/storage/supple_mic_gain` (`data_micgain.py:33`) | **shared NFS — slow** |
| `FORWARD_SEQ_ROOT`, `VIZ_RGB_ROOT` | under `/root/storage` | shared NFS |
| `ECHODIFF_PY` | `/root/local1/changwoo/echodiff_env/bin/python` | separate env (torch 1.13 + mmcv) |
| `REPLICA_ROT30_ROOT` | **absent on this machine** | `analysis/rot30.py` cannot run here |
| `R0422_SPLIT` | `off3` | scene-disjoint 12 / 3 / 3, 4800 / 1200 / 1200 samples |

`/root/storage` is the shared NFS (`/data/changwoo` on the host) and is near full; write outputs
to `/root/local1`.

## 4. Traps in this repo

- **`--epochs` is the LR horizon.** `train_oaa.py:133`: `total = a.epochs * steps_per_ep`, with
  `steps_per_ep = ceil(len(tr) / accum)` counting optimizer steps. Changing the epoch budget, or
  changing `--accum` without holding `batch-size × accum`, changes the schedule and breaks the
  comparison. To look early, keep the budget and stop the process.
- **"mic count" means heading count.** `data_micgain.py:14-17` and `:45`: 12 binaural headings on
  a 30° grid, each contributing an (L, R) pair, so channels = 2 × headings and `N_max = 24`.
  Headings are added in the bisection order in `mic_order.json`
  (0, 180, 90, 270, 30, 210, 120, 300, 60, 240, 150, 330); the first *k* entries are the
  *k*-heading set. A "12-mic" row is 24 channels.
- **Never borrow a `replica_0422` baseline into a micgain curve.** `data_micgain.py:7-12`: the
  renderer has no RNG seed, so this set's yaw 0 is *not* bit-identical to `replica_0422`'s rot-0
  wav (waveform correlation 0.82-0.98). A borrowed baseline mixes the microphone-count effect
  with renderer noise. That is the whole reason `data_micgain.py` exists as a separate module.
- **`--subset-aug` is only meaningful above 4 observations** (`train_oaa.py:7`; the drop pool at
  `:171` is built from `--vdrop-kstep`/`--vdrop-kmax`). Passing it on a 2-observation run is a
  different recipe, not the same one with a flag.
- **Checkpoint contract is strict.** `<run>/best.pth = {"state_dict", "args"}`, and OAA stores
  **EMA** weights. `core.ckpt.build(args, DM)` must rebuild from `args` alone. Guessing the key
  or the architecture silently scores EMA weights against a mismatched model and still prints a
  plausible number.
- **EchoDiffusion is a separate environment** (`requirements-echodiffusion.txt`, torch 1.13 +
  mmcv 1.7.1, float32 — the vendored SD-UNet is autocast-incompatible) and needs `HF_HOME`
  pointing at a cache holding `facebook/wav2vec2-base-960h`.

## 5. Baselines and where the citable numbers are

Citable files, both produced by `eval.py` from the `*_fin` checkpoints:

- **`comparison/compare.json`** — Replica `off3` test.
- **`comparison_mp3d/compare.json`** — Matterport3D test.
- **`comparison/fin_summary.json`** — the protocol: which seed each `_fin` run is, the GPU, the
  precision, the split and sample counts, and the note that the rn/vit/es/bat blocks were
  re-measured from the `_fin` checkpoints on 2026-08-19. Read this before comparing rows.

Verified anchors read from `comparison/compare.json` on 2026-10-13 (cos-latitude-weighted,
per-image, 256×512, Replica `off3` test):

| run | MAE (m) |
|---|---|
| `oaa_r8_fin` | 0.23682 |
| `oaa_r6_fin` | 0.23837 |
| `oaa_fb_fin` | 0.25958 |
| `oaa_r2_fin` | 0.28940 |

Non-selected seeds are in `comparison/nonselected/`; superseded runs in
`comparison/deprecated/`. Do not mix those into a current table.

**`0820_RESULTS.md`, `0820_CAMPAIGN.md` and `0820_report/` are `[문서 주장]`.** Those tables mix
intermediate settings, so a row's *name* does not establish which model and recipe produced it.
Before citing any number from them, read the run's own `train_done.json["args"]`. Any number
that cannot be found in a result file on this machine is **"기준값 없음"**, not a baseline.

## 6. Where results go

- Experiment results stay **out of the repository**: `/root/local1/changwoo/<experiment>/` with
  a `REPORT_<experiment>.md` plus `results/*.json`. The vault transfer is Changwoo's.
- Experiment **code** goes to `experiments/<name>/` (currently `e139`, `e143_e145`, `e146`,
  `e146rep`): `README.md` + `*.py` / `*.sh` / `*.diff` only. No `PREREG_*.md`, no `REPORT_*.md`,
  no figures, no `*.bak-*`, no `__pycache__`.
- `tools/make_*_tex.py` output (`comparison*/results_*.tex`) is git-ignored.

## 7. Working tree and pushing

The checkout on this node is HEAD for node work, and it currently carries **uncommitted
changes** (`0820_train_oaa_afm.py` modified, several `.bak-pre-*` untracked). Commit only the
files your change touches; never sweep that set into an unrelated commit.

The container has **no GitHub credentials** — `git push` fails on "could not read Username", and
`git ls-remote` succeeding is anonymous read, not push access. Route: bundle on the node, push
from the macmini clone at `~/work/oaa-push/oaa-audio-depth`. Never force-push, and never commit
datasets, checkpoints or experiment results.
