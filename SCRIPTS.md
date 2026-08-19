# Script index

Every runnable file, what it is for, what it reads/writes, and which reported result it backs.
Conventions: `DM` = data module chosen by `DATA_MODULE` (`data_0422` Replica / `data_mp3d` Matterport3D);
`ECO` = must run in the EchoDiffusion env (`requirements-echodiffusion.txt`); run dirs live in
`comparison/` (Replica) and `comparison_mp3d/` (MP3D). Scripts under `analysis/`, `viz/`, `tools/`
`chdir` to the repo root at start, so relative `comparison/...` paths always resolve from the root.
Scripts depend only on `core/` and `model/` — never on each other.

## Library (imported, not run)

| path | contents |
|---|---|
| `core/data.py` | `get_data_module()` — resolves `$DATA_MODULE` |
| `core/metrics.py` | `cos_lat`, `KEYS`, `BANDS`, `MetricAccumulator` (cos-lat-weighted per-image MAE/RMSE/AbsRel/log10/δ1-3 + near/mid/far) |
| `core/ckpt.py` | `build(args, DM)` rebuild any base-env model from saved args; `resolve_run`, `load_run` |
| `core/evaluate.py` | `evaluate(run_dir, DM, ckpt, device)` → metric dict |
| `model/oaa.py` | `OAAv2Depth` — the model |
| `model/batvision.py` `pretrained.py` `echoscan.py` `beyond_i2d.py` | base-env baselines (`model/__init__.py` exports them) |
| `model/echodiffusion.py` + `echodiffusion_src/` | EchoDiffusion baseline (ECO; vendored third-party code, see NOTICE.md) |
| `data_0422.py` / `data_mp3d.py` | datasets: `loader / wave_loader / spec_wave_loader`, `IN_CH`, `POSES`, `RotSet`; on-the-fly STFT |
| `splits/mp3d/*_keys.json` | MP3D scene-disjoint split (72/9/9 scenes) |

## Entry points (repo root)

| script | purpose | reads | writes | result |
|---|---|---|---|---|
| `train_oaa.py` | train OAA (`--nviews 2/4/6/8`, token masking, module/cue ablation flags, DDP, `--resume`) | DM | `<out>/<run>/{best,last}.pth, train_done.json` | Table 1 "Ours", Table 2 (`--no-*`), Table 3 retraining rows (`--pose-blind/--ear-blind`) |
| `train_batvision.py` | train BatVision (`--mode r2/cB/r6/r8`, lr 2e-3) | DM | same | Table 1 BatVision |
| `train_baseline.py` | train ResNet* / ViT* / EchoScan‡ / Beyond (`--model`, `--mode`) | DM | same | Table 1 ResNet/ViT/EchoScan |
| `train_echodiffusion.py` (ECO) | train EchoDiffusion (`--mode`, `--wave-mode std/none`, optional `--cide-cache`) | DM + wav2vec2 | same | Table 1 EchoDiffusion |
| `eval.py` | test metrics for any base-env run(s) → `compare.json` | run dirs | `<compare-dir>/compare.json` | Table 1 (non-eco cells) |
| `eval_echodiffusion.py` (ECO) | same for EchoDiffusion runs | run dirs | `<compare-dir>/compare_eco.json` | Table 1 eco cells |

## analysis/ — numbers beyond Table 1

| script | what it measures | ckpt(s) | writes | result |
|---|---|---|---|---|
| `blind.py` | pose cue removed at inference only | OAA run(s) | `comparison/eval_blind.json` | Table 3 (inference rows) |
| `eardrop.py` | one ear's channels zeroed at inference (poses truthful) | OAA r8 run(s) | `comparison/eval_eardrop.json` | Table 3 (ear cue) |
| `controls.py` | audio-shuffle / pose-shuffle / L-R-swap controls (no retraining) | OAA runs | `comparison/controls.json` | Table 3 support |
| `micdrop.py` | progressive k-of-8 observation drop, fixed random subsets | OAA r8 run | `comparison/eval_micdrop.json`, `fig_missing_mics*.{png,pdf}` | Table 4, supp. Table 2 |
| `micdrop_eco.py` (ECO) | same protocol for EchoDiffusion (dropped front mic also drops its waveform) | eco r8 run | `comparison/eval_micdrop.json` | Table 4, supp. Table 2 |
| `rot30.py` | unseen receiver headings (rig rotated 30°/45° multiples; needs `$REPLICA_ROT30_ROOT`) | `--model oaa/bat/eco --ckpt` | `comparison/rot30_eval_*.json` | Table 5, supp. Table 1 |
| `rot_jitter.py` | per-mic heading jitter (true vs nominal poses for OAA) | `--model --ckpt` | `comparison/rotjitter_eval_*.json` | Table 5 (3rd row), supp. Table 1 |

## viz/ — figures

| script | figure | notes |
|---|---|---|
| `grid.py` / `grid_mp3d.py` | Fig. 4 qualitative grid (2 samples per test scene, columns GT/ResNet/ViT/(Beyond blank)/EchoScan/BatVision/EchoDiffusion/Ours) | stage 1 predicts (base env), `eco.py` adds eco predictions (ECO), then `--compose` |
| `full.py` / `full_mp3d.py` | supp. Fig. 2-3 — 1×8 strip for every test sample | eco predictions from `eco_full*.py` memmaps |
| `full_indiv.py` / `full_indiv_mp3d.py` / `full_hires.py` | per-model native-resolution PNGs / hi-res strips | |
| `eco.py` `eco_mp3d.py` `eco_full.py` `eco_full_mp3d.py` (ECO) | EchoDiffusion predictions as npy for the scripts above | |
| `mic_attr.py` | Fig. 5 — observation attribution (occlusion, keep-only, RayMicAttn attention mass per ERP region) | writes `comparison/mic_attribution/` |
| `mic_attr_eco.py` / `gradattr_eco.py` (ECO) | Fig. 5 EchoDiffusion counterpart (grad×input) | |
| `forward_seq.py` | supp. HTML appendix — trajectory sequences (`$FORWARD_SEQ_ROOT` → `$FORWARD_SEQ_OUT`) | |

## tools/

| script | purpose |
|---|---|
| `make_replica_tex.py` / `make_mp3d_tex.py` | `compare*.json` → LaTeX result tables (`comparison*/results_*.tex`, git-ignored) |
| `export_configs.py` | dump `train_done.json["args"]` of the final runs → `configs/{replica,mp3d}/<run>.json` |
| `build_spec_cache_replica.py` / `build_spec_cache_mp3d.py` | optional bit-identical STFT caches (`$REPLICA_SPEC_CACHE` / `$MP3D_SPEC_CACHE`) — speed only |
| `build_cide_cache.py` (ECO) | precompute wav2vec2 CIDE embeddings (`cache_cide/`) for `train_echodiffusion.py --cide-cache` |

## Environment variables

`DATA_MODULE`, `REPLICA_ROOT`, `MP3D_ROOT`, `MP3D_KEYS` (default `splits/mp3d`), `R0422_SPLIT` (default `off3`),
`EVAL_BS`, `REPLICA_SPEC_CACHE`, `MP3D_SPEC_CACHE`, `REPLICA_ROT30_ROOT`, `FORWARD_SEQ_ROOT`, `FORWARD_SEQ_OUT`,
`VIZ_RGB_ROOT`, `HF_HOME` / `ECHODIFF_WEIGHTS` (ECO). A local, git-ignored `env.local.sh` can hold the machine's values.

## Checkpoint / run-dir contract

`<run>/best.pth = {"state_dict", "args"}`; `core.ckpt.build(args, DM)` must be able to rebuild the model from
`args` alone (trainers write every architecture-relevant flag into `args`, plus `data_module`). OAA saves EMA weights.
`train_done.json` = `{"best_val_mae_m", "hist": [...], "args"}`. Final runs: `*_fin` (see `comparison_mp3d/FIN_SELECTION.md`).
