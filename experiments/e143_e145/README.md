# experiments/e143_e145 — EchoRecon OAA trainers E143 → E145

These files ran on the UNIST lab nodes out of `/root/storage/e14?_code/`, which are **not** git
checkouts, so until this commit the crash-safe trainer existed only on the nodes. Committed here as
**code only**: no datasets, no checkpoints, no experiment results, no pre-registrations or reports.

## The patch chain (each step is checkable, not just claimed)

`train_oaa.py` (repo root) → `train_oaa_e135.py` → `train_oaa_e143.py` → `train_oaa_e144.py` →
`train_oaa_e145.py`.

The last two steps are produced by generator scripts rather than hand edits. Each generator performs
literal string replacements, **asserts every replacement matches exactly once**, and refuses to run
if the source md5 differs — which is what makes "everything else is byte-identical" verifiable.

| step | generator | edits | source md5 | output md5 |
|---|---|---|---|---|
| E143 → E144 | `e144_patch.py` | 5 | `2926c74210175dc03d9dc5591bb76566` | `df08858a74fa39f7c7c9bf4c42427a6f` |
| E144 → E145 | `e145_patch.py` | 7 | `df08858a74fa39f7c7c9bf4c42427a6f` | `39debb9beb863a1bde02316cbce8670c` |

Reuse this pattern for future trainer patches instead of editing a trainer in place.

### E144: crash-safe checkpointing (no recipe change)

`train_oaa_e143.py` and every earlier OAA trainer wrote `last.pth` **once, after the final epoch**,
so an 80-epoch run that died at epoch 60 was unresumable. E144 adds:

- `atomic_save()` — tmp file → `fsync` → `os.replace`, so a kill mid-write cannot truncate the file.
- `last.pth` written **every epoch, inside the loop**, carrying `best`, `hist` and RNG state
  (torch / cuda / numpy / python) so dying before `train_done.json` no longer resets model selection.
- `--resume auto` — resolves to this run dir's own `last.pth`, and exits 0 cleanly when the run is
  already at `--epochs` (E143 died on an assert there).

Fresh runs are numerically identical to E143: the only added calls are `torch.save` plus RNG
*getters*, and getters do not advance a generator. Resume was verified not to cost an epoch
(1 → 3 epochs, `hist` continuous, log printed `rng=restored`).

### E145: per-arm objective-aware model selection (no recipe change)

E144 selected `best.pth` on `quick_val`'s **unweighted** cos-latitude masked L1, which is not the
quantity a `--loss-weight`-reweighted run minimises. E145 records both numbers every epoch and saves
both checkpoints from **one** training run:

| checkpoint | selected on |
|---|---|
| `best.pth` | val unweighted MAE — E144's metric verbatim |
| `best_own.pth` | val MAE with the arm's own training distance weight |

With `--loss-weight none` the weight function returns exactly `1.0`, so the two metrics are
bit-identical and the two checkpoints hold identical content. That equality is the integrity check.

`val_trainform` is also logged (training-loss formula verbatim: normalised depth, no cos-latitude).
It is **diagnostic only and selects nothing** — it exists to measure whether dropping the
cos-latitude term would name a different epoch.

The training step, optimizer, cosine schedule, EMA, loss and RNG consumption are untouched:
`quick_val` runs under `no_grad` in `eval()` mode and only adds arithmetic on tensors it already
holds.

### Comparing two checkpoints — use content, not file bytes

`torch.save` names the zip container after the **output filename**, so `best.pth` holds entries
`best.pth/data.pkl …` and `best_own.pth` holds `best_own.pth/data.pkl …`. Two checkpoints carrying
identical weights are therefore **never** byte-identical; measured on the E145 smoke run as 304/304
state_dict tensors equal, `args` equal, file sha256 different. `ckpt_content_hash.py` hashes
`state_dict` (keys sorted, with dtype/shape/values) plus `args` instead, and is what the integrity
check uses.

## Runners

`run_worker_e14?.sh` (one GPU, queue of runs, `--resume auto`, logs **appended** with `>>` — `>` is
what permanently destroyed E143's training logs), `launch_e14?.sh` (one tmux window per training run
plus one scorer window per run, no `nohup`), `score_e14?.sh` (shell-polls
`until [ -f train_done.json ]`, then predicts and evaluates), `finish_e14?.sh` (waits for all cells,
aggregates), `status_e145.sh` (machine-parseable status probe, invoked **by path** so a watcher never
needs nested quoting), `smoke_e145.sh` + `check_smoke_e145.py` (prove the two-checkpoint mechanism
before a long launch), `aggregate_e14?.py` (pre-registered thresholds hardcoded).

## Node-specific paths and the two things deliberately NOT committed

Paths inside these scripts are lab-node paths (`/root/storage/...`, `/root/local1/...`) and are kept
verbatim so the commit records what actually ran. Adjust them before reuse elsewhere.

Not committed, on purpose:

1. **`gate0_bin_dist.json`** — the per-range-bin training-set distribution that
   `--loss-weight invfreq` reads (`E143_BIN_DIST`, default `/root/storage/e143_code/gate0_bin_dist.json`).
   It is a dataset statistic, and datasets stay out of the repo. Values used by every E143–E145
   invfreq run, recorded here so the arm is reproducible without shipping the file:
   `bin_edges = [0.5, 1.5, 4.0]`, `invfreq_weight_train = [3.8910277266323696, 0.5222029016181182,
   0.5824784549622878, 8.99019209631278]`, `mean_sqrt_d_train = 1.1824380159378052`.
2. **The eval/predict chain** — `e114_eval.py`, `e143_predict.py`, `list_seqs.py`. These import from
   the sibling **EchoRecon** repo's `src/`, so they belong to that repo, not to this one; filing them
   here would put them in the wrong place. They still live only on the nodes in
   `/root/storage/e143_code/`. Metric code used by E143–E145 is `e114_eval.py` md5
   `5e28675599c7872e34258a1038f503c8`, unmodified throughout.

Pre-registrations, reports and result JSONs are experiment record, not code, and stay under
`/root/local1/changwoo/e14?/` for transfer into the vault.
