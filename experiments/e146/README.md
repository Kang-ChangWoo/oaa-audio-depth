# E146 — radial loss bins

The distance-reweighted loss used by E114 / E143 / E144 / E145 binned **planar** depth
(per-face cubemap z-depth, what `erp_depth/` stores), while `e114_eval.py` bins the **radial**
range (`EchoRecon/src/erp.py`: `to_radial(d, dirs, "face") = d / face_cos`). The treated pixel set
and the measured pixel set were different sets. E146 moves both the loss bins and the inverse
frequency histogram into radial metres so they agree, and asks whether `invfreq`'s `>4 m` gain grows.

6 runs: `ctrl` x seeds {0,1,2} (reproduction anchor — `--loss-weight none` makes the treatment a
mathematical no-op) and `invfreq` x seeds {0,1,2} (the binding arm). 80 epochs, cosine 80, exactly
E144/E145's budget.

**`bin4` is deliberately NOT re-run.** `CUE_CH = {"none":1,"ild":2,"ipd":3,"bin4":4}` and
`data_0422_bin.py`'s layout is `[mag, ILD, IPDc, IPDs]` — "bin4" means *binaural 4 channels* of
audio input, not distance bins, and its `--loss-weight` is `none`. The radial move cannot touch it.

## Files

| file | what |
|---|---|
| `e146_patch.py` | builds `train_oaa_e146.py` from `train_oaa_e145.py` by 5 literal replacements, each asserted to match exactly once; base md5 checked first |
| `e146_patch.diff` | the resulting diff: 5 hunks, 4 removed lines |
| `train_oaa_e146.py` | the trainer actually run (md5 `a2cf6b622698db872a9b7cd2c4ffc9ec`) |
| `gate0_bin_dist_radial.py` | E114's `gate0_bin_dist.py` with `depth /= face_cos`; reports native 512x1024 and trainer-grid 256x512, planar and radial |
| `check_bins_e146_full.py` | integrity check: walks the whole training index through the trainer's own `_load_depth` and proves the bins moved |
| `check_bins_e146.py` | the first, sampling-biased version of that check — kept because the report cites why it failed |
| `aggregate_e146.py` | gate G1 + the UP/DOWN/TIE/TRADE landing, PREREG constants hardcoded |
| `agg_n_gate_patch.py` | adds the `n != 3` gatekeeper to the E144/E145 aggregators |
| `run_worker_*.sh` `launch_*.sh` `score_*.sh` `finish_*.sh` | queue / scorer / finisher, all shell `until` loops, no LLM polling |
| `e146_probe.sh` | one-line node status for the watcher; a FILE so the watcher never nests shell quoting |

## What the patch changes

1. loads EchoRecon's `erp.py` **by file path** (`importlib`), not via `sys.path.insert` —
   `EchoRecon/src` also holds `data.py` / `predict.py` / `eval_fusion.py`, which could shadow a
   hear360 module later in the run
2. `BIN_DIST_PATH` -> `gate0_bin_dist_radial.json`
3. training loop: `d_m = (gt * max_depth / face_cos).clamp(0.05, max_depth)` —
   **the loss target stays planar `gt`**, only the per-pixel weight moves
4. `quick_val`'s own-objective weight moves to the same space
5. the banner prints `bin_space=radial` and the weight file, so every run log proves which it ran

Unchanged: loss target, optimizer, cosine schedule, EMA, split, augmentation, RNG consumption,
`val_mae_m` (which selects `best.pth` and carries no distance weight at all — that is what makes the
reproduction gate a real test), checkpoint format, `--resume auto`.

## Dataset statistic (not committed — `gate0_bin_dist_radial.json` is data, values reproduced here)

Train split, 12 scenes, 4800 frames. `gate0_bin_dist_radial.py`'s planar output reproduces E114's
`gate0_bin_dist.json` exactly (`max|diff| = 0.0`), which is what makes this a single-variable change.

| bin | planar frac | planar w | radial frac | radial w |
|---|---|---|---|---|
| `<0.5` | 6.4250 % | 3.8910 | 4.1563 % | 6.0150 |
| `0.5-1.5` | 47.8741 % | 0.5222 | 39.6846 % | 0.6300 |
| `1.5-4` | 42.9200 % | 0.5825 | 52.0144 % | 0.4806 |
| `>4` | 2.7808 % | 8.9902 | 4.1448 % | 6.0317 |

32.91 % of the truly-`>4 m` pixels were being binned as `1.5-4` by the planar expression.

`1/face_cos` on the 256x512 grid runs 1.00002 .. 1.72172, and `face_cos` is convention-invariant
(`max|diff| = 4.4e-16` across all four `CONVENTIONS`), so using the eval's `"right0"` is unambiguous.
`data_0422._load_depth`'s `F.interpolate(mode="nearest")` 512x1024 -> 256x512 was verified to equal
`d[::2, ::2]` exactly, and `e114_eval.py` builds its `dirs` at the same 256x512 prediction grid.

## Note on the invfreq weights

`bin_w = inv / inv.dot(frac)` is an identity that forces `frac[i] * bin_w[i] = 1/4`: **every bin
always receives exactly 25 % of the loss mass, whatever the bin definition.** Measured mass is
0.2500 in every grid. So E146 is not "push the far range harder" — it is "give that same 25 % to a
more accurate set of pixels". The `>4 m` weight goes *down* (8.99 -> 6.03) while the bin's population
goes up (2.78 % -> 4.14 %).

## `n != 3` gatekeeper

`ms()` / `sd()` in the E144 and E145 aggregators compute a **population** standard deviation, so a
cell with one surviving seed reports sigma = 0.0 — and the pre-registered rule "|delta| < pooled
sigma => TIE" then makes every difference, however small, decisive. The decision rule is
pre-registered and is **not** modified. `agg_n_gate_patch.py` only adds a gate: a comparison whose
either side is not a full 3-seed cell is reported `WITHHELD_N` instead of decided.

Verified regression-free: re-running the patched `aggregate_e145.py` reproduces the stored
`e145_agg.json` byte-for-byte (verdict `HOLD`), and `aggregate_e144.py` keeps verdict `HOLD` with
both orderings unchanged and no `WITHHELD`.

Results (PREREG / REPORT / `results/*.json`) stay on the node under
`/root/local1/changwoo/e146/` and are not committed.
