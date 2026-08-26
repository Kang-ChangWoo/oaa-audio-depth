# 0820 Audio-Foundation-Model encoder experiments — results (2026-08-20 ~ 08-23)

Replace ONLY the coarse per-observation encoder of OAA (HEAR 360°) with pretrained ViT-B/16 audio
foundation models; everything else (fine CNN skip, pose/ear/TF embeddings, alternating + ray-mic
geometric attention, decoder, loss, recipe) unchanged. 4-obs (fb) screening on Replica + MP3D,
then hypothesis-testing variants. 21 runs total; run dirs in `comparison_0820/` (not versioned).
References: `comparison/oaa_fb_fin` (Replica) / `comparison_mp3d/oaa_fb_fin` (MP3D). Seed 0 unless noted.

## Headline

**`eat + LLRD(0.75) + warmup8` beats the paper CNN on MP3D** — the first AFM to do so:
test MAE 0.773 vs 0.785 (−1.5%), RMSE 1.257 vs 1.276, δ1 0.541 vs 0.538 (val 0.8912 vs 0.8920).
On Replica, `sslam` ties/edges the CNN (test MAE 0.255 vs 0.260; mid/far-band −5%) with default recipe.

## Stage 1 — screening (4ch fb, identical recipe, AFM LR = 0.1x)

| val best MAE | Replica (CNN 0.2905) | MP3D (CNN 0.8920) |
|---|---|---|
| sslam | **0.2924** | 0.9755 (LR spike) |
| audiomosaic | 0.3116 | 0.9326 |
| eat | 0.3129 | 1.0088 (LR spike) |
| m2d (CLAP'25 16x16) | 0.3146 | 0.9242 |
| bat | 0.3281 | 0.9240 |

No backbone beat the CNN on either dataset with the plain recipe; per-dataset ranking flipped
(sslam best on Replica, bat/m2d best on MP3D). Post-norm models (eat/sslam) showed LR-peak val
spikes on MP3D. Learning-curve overlap with the CNN on Replica suggested the encoder is not the
bottleneck there.

## Stage 2 — hypothesis-testing variants (verdicts)

| hypothesis | run(s) | result | verdict |
|---|---|---|---|
| LR recipe (LLRD 0.75 + warmup 8) fixes post-norm spikes | eat_llrd / sslam_llrd (MP3D) | eat 1.0088→**0.8912** (< CNN), sslam 0.9755→0.9152; no spikes | **VALID — the key fix** |
| pretraining matters (vs architecture) | sslam_randinit (Rep) | 0.3781 vs 0.2924 (+29%) | **VALID** (pretraining contributes a lot) |
| 16x16 patch destroys ToF time resolution | m2d20ms = M2D-CLAP 80x2, patch (128,2) | MP3D 0.9242→**0.9136**; Rep 0.3164 (mid) | **PARTIAL** (helps MP3D, not enough alone) |
| sub-patch locality (conv patch stem) | sslam_convstem (Rep) | 0.3017 vs 0.2924 | rejected |
| input-statistics matching (dB / dB-minmax) | bat_dbmm, audiomosaic_db | 0.417/0.431 Rep, 1.047 MP3D — much worse | **rejected** (log1p+std is right) |
| seed noise scale | sslam_s1 | 0.3047 vs s0 0.2924 → spread ≈ 0.012 | sslam-vs-CNN Replica gap (0.002) = tie |

## Final 4ch test tables

MP3D (3600 samples): CNN 0.785/1.276/0.376/0.538 (MAE/RMSE/AbsRel/δ1)
→ **eat_llrd 0.773/1.257/0.378/0.541**; sslam_llrd 0.792/1.273; m2d20ms 0.796/1.279; others worse.

Replica (1200): CNN 0.260/0.514/0.140/0.826
→ **sslam 0.255/0.502/0.142/0.826** (mid3-6m 0.715 vs 0.758, far>6m 1.446 vs 1.520; near<3m worse
0.142 vs 0.135); sslam_llrd 0.258/0.510; convstem 0.259/0.511; others worse.

## Interpretation

1. AudioSet pretraining transfers (randinit +29% worse), but the plain 0.1x uniform-LR recipe
   wastes it on post-norm models; LLRD unlocks it. 2. AFM gains concentrate in mid/far range
   (late-reverb cues); the CNN stays better near-field — time-resolution (20ms patch) narrows but
   does not close that gap. 3. dB compression of echo magnitudes is harmful. 4. The 15.4M CNN
   remains remarkably strong per-parameter (AFMs are 99-106M).

## Recommended next steps

1. **eat_llrd 2/6/8ch on both datasets** (stage 3, the winner) + Replica eat_llrd (fb) for the
   both-datasets claim; multi-seed (s1/s2) for the MP3D win.
2. sslam_llrd MP3D longer/LLRD-tuned (0.9152 with spike removed still trails eat).
3. Optional: m2d20ms + LLRD combination (time-resolution + recipe), DAPT (continued SSL
   pretraining on echo spectrograms) as the deeper fix for the near-field gap.

Commands: see `0820_queue_screen.sh`, `0820_queue_stage2.sh`; per-run args in each
`comparison_0820/<run>/train_done.json`.

## Stage 3 — eat_llrd multi-seed + channel scaling (2026-08-24 ~ 08-26, runs `0820_queue_stage3.sh` / `_stage3b.sh`)

All rows `eat + LLRD 0.75 + warmup 8`, seed 0 unless noted. Test MAE (m); CNN = paper finals.

**MP3D fb multi-seed — the win replicates (3/3 seeds):**
s0 0.7728 / s1 0.7740 / s2 0.7684 -> mean **0.7717 ± 0.0024** vs CNN 0.785 (−1.7%); every seed < CNN.
Variant `--epochs 30` (faster cosine) 0.7689 — slightly better than 40ep s0, verdict VALID as an
equal-cost alternative; m2d_llrd 0.8160 and m2d20ms_llrd 0.8082 — both REJECTED (LLRD does not
rescue the m2d family; the backbone matters, not just the recipe).

**Channel scaling (test MAE, AFM vs CNN):**

| obs | Replica AFM | Replica CNN | MP3D AFM | MP3D CNN |
|---|---|---|---|---|
| r2 (2ch) | **0.2762** | 0.2894 | **0.9018** | 0.9084 |
| fb (4ch) | 0.2609 | 0.2596 | **0.7717±0.0024** | 0.785 |
| r6 (6ch) | 0.2427 | 0.2384 | (running) | 0.7502 |
| r8 (8ch) | 0.2371 (RMSE 0.4712<0.4810) | 0.2368 | (running) | 0.7467 |

Note val->test rank flips are common (fb Rep val 0.3085 -> test 0.2609 ≈ CNN; r8 Rep val 0.2752
vs CNN 0.266 -> test tie; r2 MP3D val 1.048 vs 0.908 -> test WIN): compare on test, not val.

**Interpretation update:** the AFM advantage is largest where observations are scarcest (r2: wins
on BOTH datasets, −4.6% / −0.7%) and on the harder/noisier dataset at any channel count (MP3D fb
−1.7% across 3 seeds). With rich observations on clean Replica the CNN's near-field precision
keeps it level or slightly ahead (r6/r8 ties). AudioSet features buy robustness, not precision.
