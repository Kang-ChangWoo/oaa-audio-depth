# comparison/ (Replica) and comparison_mp3d/ (Matterport3D) — run directories and result tables

```
<run-name>/best.pth         {"state_dict": weights (EMA for OAA), "args": {...}}
<run-name>/last.pth         resumable bundle (raw weights + optimizer + scheduler)
<run-name>/train_done.json  best val MAE, per-epoch history, full args
compare.json                eval.py summary (all base-env models)      compare_eco.json  eval_echodiffusion.py summary
fin_summary.json            which seed/run was selected per cell + environment notes (Replica)
FIN_SELECTION.md            Matterport3D selection notes (comparison_mp3d/)
controls.json, eval_blind.json, eval_eardrop.json, eval_micdrop.json, rot30_eval_*.json, rotjitter_eval_*.json
                            analysis outputs (see the eval_*.py docstrings)
```
Run directories and checkpoints are not versioned (see .gitignore); only the summary files are.
Final-selection runs are `*_fin` on Replica; on Matterport3D the EchoDiffusion finals are
`eco_r2_wstd`, `eco_fb_wstd`, `eco_r6`, `eco_r8` (see FIN_SELECTION.md). `deprecated/` and `nonselected/`
hold superseded runs.

Train / evaluate into these folders with `--out-dir comparison` (Replica) or `--out-dir comparison_mp3d`
(Matterport3D); `eval.py --compare-dir <dir>` writes the table there.
