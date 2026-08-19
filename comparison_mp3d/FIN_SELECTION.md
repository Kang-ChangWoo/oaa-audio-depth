# Matterport3D final-run selection (fixed 2026-07-27)

Source runs of the Matterport3D rows of the results table. EchoDiffusion finals = **wstd** (the upstream
default wave setting; wall/wnone variants are kept in nonselected/ for ablation footnotes). Ours = the
full-resolution multi-scale model throughout (15.4 M params).

| table row | final dir | original run name (old key may coexist in compare.json) | MAE (m) |
|---|---|---|---|
| 2 obs. Ours | `oaa_r2_fin` | oaa_r2 (bs16, 40 ep) | 0.9084 |
| 4 obs. Ours | `oaa_fb_fin` | oaa_fbm_fr_s0 (fb channels, eff. batch 32, 40 ep) | 0.7849 |
| 6 obs. Ours | `oaa_r6_fin` | oaa_r6m_s1_s0 (+stem_stride1, eff. batch 16, 30 ep) | 0.7502 |
| 8 obs. Ours | `oaa_r8_fin` | oaa_r8m_s1kany_s0 (+stem_stride1, token masking k-any) | 0.7467 |
| 2 obs. eco | `eco_r2_wstd` | — | 0.9007 |
| 4 obs. eco | `eco_fb_wstd` | — | 0.7928 |
| 6 obs. eco | `eco_r6` (wstd) | — | 0.7786 |
| 8 obs. eco | `eco_r8` (wstd) | — | 0.7572 |
| baselines | `{bat,vit,rn,es}_{r2,fb,r6,r8}_fin` | rn_fb / rn_r2 / vit_r2 were renamed from un-suffixed runs | — |

- Run directories were renamed after selection, so `train_done.json["args"]["run_name"]` differs from the
  directory name; `configs/mp3d/*.json` records the exact arguments of each final run.
- All non-selected runs are under `nonselected/`, superseded ones under `deprecated/`.
