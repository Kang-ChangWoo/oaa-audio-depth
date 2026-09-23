# Final benchmark — current best AFM candidate across all eight cells

Model: **sslam + LLRD + conv stem, STFT hop 44**, one fixed configuration, single seed
unless stated. Numbers are read from the campaign's compare.json files; the baseline run
names resolved for each cell are listed at the end so the mapping is auditable.

Tie rule: |dMAE| < 0.01 is a tie (the campaign's existing threshold).

## Results (test MAE, metres)

| Dataset | Ch | EchoDiffusion | OAA-CNN | SSLAM+LLRD+CS+h44 | vs CNN | vs Eco |
|---|---:|---:|---:|---:|---:|---:|
| Replica | 2 | 0.2854 | 0.2894 | **0.2638** | -0.0256 | -0.0216 |
| Replica | 4 | 0.2695 | 0.2596 | **0.2496** | -0.0100 | -0.0199 |
| Replica | 6 | 0.2556 | 0.2384 | **0.2353** | -0.0031 | -0.0203 |
| Replica | 8 | 0.2600 | 0.2368 | **0.2371** | +0.0003 | -0.0229 |
| MP3D | 2 | 0.9007 | 0.9084 | **0.8836** | -0.0248 | -0.0171 |
| MP3D | 4 | 0.7928 | 0.7849 | **0.7562** | -0.0288 | -0.0367 |
| MP3D | 6 | 0.7786 | 0.7502 | **0.7379** | -0.0123 | -0.0407 |
| MP3D | 8 | 0.7572 | 0.7467 | **0.7275** | -0.0192 | -0.0297 |

## Aggregates

- all mean MAE: 0.5114 (8 cells)
- Replica mean MAE: 0.2464 (4 cells)
- MP3D mean MAE: 0.7763 (4 cells)

- vs OAA-CNN: **6 win / 2 tie / 0 loss**
- vs EchoDiffusion: **8 win / 0 tie / 0 loss**

## Matched-hop control (every model at hop 44)

The table above reads our hop-44 model against baselines trained at hop 160, which prices
the input recipe together with the encoder. This block retrains both baselines on the same
input; cells still training read pending.

| Dataset | Ch | OAA-CNN @44 | EchoDiffusion @44 | ours @44 | vs CNN@44 | vs Eco@44 |
|---|---:|---:|---:|---:|---:|---:|
| Replica | 2 | 0.2744 | pending | **0.2638** | -0.0106 | — |
| Replica | 4 | 0.2496 | pending | **0.2496** | +0.0000 | — |
| Replica | 6 | pending | pending | **0.2353** | — | — |
| Replica | 8 | 0.2350 | pending | **0.2371** | +0.0021 | — |
| MP3D | 2 | pending | pending | **0.8836** | — | — |
| MP3D | 4 | 0.7744 | pending | **0.7562** | -0.0182 | — |
| MP3D | 6 | pending | pending | **0.7379** | — | — |
| MP3D | 8 | 0.7365 | pending | **0.7275** | -0.0090 | — |

- matched-hop vs OAA-CNN: **2 win / 3 tie / 0 loss** (5 of 8 cells measured)
- matched-hop vs EchoDiffusion: **0 win / 0 tie / 0 loss** (0 of 8 cells measured)

## Baseline run names resolved per cell

| cell | OAA-CNN | EchoDiffusion | ours |
|---|---|---|---|
| Replica 2ch | `oaa_r2_fin` | `eco_r2_fin` | `0820_h44_sslcs_r2_rep` |
| Replica 4ch | `oaa_fb_fin` | `eco_fb_fin` | `0820_h44_sslcs_fb_rep` |
| Replica 6ch | `oaa_r6_fin` | `eco_r6_fin` | `0820_h44_sslcs_r6_rep` |
| Replica 8ch | `oaa_r8_fin` | `eco_r8_fin` | `0820_h44_sslcs_r8_rep` |
| MP3D 2ch | `oaa_r2` | `eco_r2_wstd` | `0820_h44_sslcs_r2_mp3d` |
| MP3D 4ch | `oaa_fbm_fr_s0` | `eco_fb_wstd` | `0820_h44_sslcs_fb_mp3d` |
| MP3D 6ch | `oaa_r6_fin` | `eco_r6` | `0820_h44_sslcs_r6_mp3d` |
| MP3D 8ch | `oaa_r8_fin` | `eco_r8` | `0820_h44_sslcs_r8_mp3d` |
