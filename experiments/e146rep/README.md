# E146REP — `ctrl` repeat-noise probe, and the watcher bug that hid it for 5 hours

Landed 2026-09-28. Results stay off the repo, under `/root/local1/changwoo/e146rep/`.

## What it measured

E146 landed `VOID` because pre-registered gate G1 failed on `>4 m` F1: the `ctrl` 3-seed mean
overshot the window's upper edge by `1.25e-5`. `PREREG_E146.md §3.1` named two candidate causes
(radial-bin treatment vs the n1→n4 node move). This probe measured a **third**: the run-to-run
noise of a same-seed, same-node repeat.

`ctrl_s2` was run once more (`ctrl_s2_rep1`, seed 2, `--cue none --loss-weight none`, 80 epochs,
n4 GPU5, trainer/evaluator/data byte-identical to E146). Pre-registration in
`PREREG_E146REP.md` (md5 `feb548e29441667ec1ab2e578d0a1617`, written before launch).

**Verdict: `DRIFT_REAL`.** σ_rep = .002161 (population sd of the three same-cell `>4 m` F1
observations .090228 / .085549 / .085744) < σ_G1 = .003905. Same answer with ddof=1 (.002647).
E146 stays `VOID`; the next step is a n1↔n4 bisect, not a new gate. `aggregate_e146rep.py`
writes `results/verdict_e146rep.json`.

### The three numbers are not commensurable

The tempting reading — "G1 missed by `1.25e-5` but reruns differ by `.0002`, so the gate was a
knife edge" — does not hold. `1.25e-5` is the **residual margin of a mean** past a window edge,
not a dispersion. `.0002` is **one sample** of single-run dispersion. `.0039` is a single-run
seed-to-seed σ used as the half-width for a **3-seed mean**.

The quantity G1 actually gated is the mean shift `+.003918 = 1.0032 σ_G1`. Expressed in repeat
noise, the E145→E146 mean shift is `+.003949 = 2.24 ×` the SE of a difference of two 3-seed means
(σ_rep/√3·√2 = .001765). Repeat noise does not explain it. Note also that scaling the gate
*correctly* for a mean (σ/√3) makes the same data fail **harder** (.00166 overshoot) — the gate
was loose, not tight.

The drift is not systematic across seeds: s0 −0.08 σ_rep, **s1 +7.73 σ_rep**, s2 −2.17 σ_rep.
The deciding observation is a single rerun of `ctrl_s1` (`REPORT_E146REP.md §2.8`).

## Why training is not bit-reproducible: FlashAttention backward

`determinism_probe*.py` (n4 GPU7, ~30 s each, no training) settled this.

The trainer sets **no** determinism switch: `cudnn.benchmark=False` (default),
`cudnn.deterministic=False`, `use_deterministic_algorithms=False`, `CUBLAS_WORKSPACE_CONFIG`
unset; `train_oaa_e146.py:278` calls only `torch.manual_seed` + `np.random.seed`.

| probe | result |
|---|---|
| `F.interpolate` bilinear bwd (`oaa.py:90`) | DETERMINISTIC — ruled out |
| `ConvTranspose2d` bwd (`oaa.py:191`) | DETERMINISTIC — ruled out |
| full `OAAv2Depth` bwd under bf16 autocast | **NON-DETERMINISTIC** |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` alone | still non-deterministic — cuBLAS ruled out |
| `cudnn.deterministic=True` alone | still non-deterministic — cuDNN conv ruled out |
| `use_deterministic_algorithms(True)` + env var | DETERMINISTIC |

Under `warn_only=True` torch emits exactly one warning: *"Flash Attention defaults to a
non-deterministic algorithm"* (`attention_backward.cu:110`). The entry points are
`nn.MultiheadAttention` throughout `model/oaa.py` (`SelfAttn`×4, `CondSelfAttn`×rounds,
`InterMicAttn`×rounds, `RayMicAttn`, `fine_lift`) reaching the torch 2.8 SDPA fast path, whose
backward accumulates dQ with atomics.

So **bit-reproducibility is architectural, not environmental.** Pinning the node cannot make
reruns reproduce; `last.pth` ep0 losses are three distinct values (0.160802912 / 0.160822606 /
0.160819012) while agreeing to 4 digits, i.e. identical init and data order with a divergent
backward. Reproduction gates must therefore be distributional, with σ measured from same-cell
repeats and scaled by √n when applied to an n-seed mean.

## The watcher bug — third of its family

`score_rep.sh` writes each scored cell to `$OUT/eval_results/<run>__<tag>.json` and *also* copies
it to `$E146CODE/collect_e146rep/`, where `E146CODE=/root/storage/e146_code` (inherited verbatim
from `score_e146.sh`). `e146rep_probe.sh` counted `$CODE/collect_e146rep/`, where
`CODE=/root/storage/e146rep_code`. The E146→E146REP rename landed on the counter but not on the
writer, so `cells` stayed `0/2` while `done=1`, and the watcher reported `state=RUNNING` for 5 h
with every artifact already on disk. E144's watcher died the same way (4 h), E146's survived only
because both literals happened to match.

**Root cause:** the writer and the counter each carry their own path literal, and nothing asserts
they agree. Two fixes, both in this directory:

1. `e146rep_probe.sh` now counts the scorer's **primary** write, `$OUT/eval_results/` — the one
   path both scripts already share. The collect dir is only a cross-node convenience copy.
2. `e146rep_trigger.sh` gained a `state=INCONSISTENT` branch: `done=1` with `cells<2` is a state
   the pipeline cannot legitimately be in, so it **fires** after two consecutive polls instead of
   staying quiet. Whatever the next path bug is, it costs one poll interval, not a morning.
   `E146REP_PROBE` is now overridable so that state machine can be exercised against a stub probe
   (verified: quiet on poll 1 exit 1, `INCONSISTENT` on poll 2 exit 0, real path still
   `DONE cells=2/2` exit 0).

## Files

| file | role |
|---|---|
| `run_worker_rep.sh` | single-GPU worker that trained `ctrl_s2_rep1` |
| `score_rep.sh` | shell-polled scorer: predict + eval both checkpoints, integrity check |
| `aggregate_e146rep.py` | applies `PREREG_E146REP.md §3` verbatim; scale audit + overall axis |
| `e146rep_probe.sh` | node-side one-line status probe (cell-count fixed) |
| `e146rep_trigger.sh` | macmini-side watcher trigger (`INCONSISTENT` guard added) |
| `determinism_probe.py` … `4.py` | isolates the non-deterministic kernel; probe 4 makes torch name it |
