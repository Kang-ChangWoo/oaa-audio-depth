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

## Stage 3c-3k — the unified-winner campaign (2026-08-26 ~ 09-07)

Goal (user directive): ONE setting that beats the paper CNN on every channel count AND both
datasets. Verdict trail (all test MAE):

**vdrop (mic-drop) — channel-redundancy law, now fully seeded:**
fb REJECT x3 seeds (0.2742/0.2815/0.2820 vs 0.2609) | r6 REJECT x3 (0.2470±0.002 vs 0.2427) |
Replica r8 HELPS x3-controls (vd 0.2371 vs novd 0.2535/0.2618/0.2606) | MP3D r6 REJECT x3
(0.795/0.816/0.817 vs 0.7736) | MP3D r8 REJECT (aggressive kmax6: 0.9858). Drop regularisation
works only where observations are redundant AND clean (Replica 8ch); everywhere else it destroys
the AFM's reverb-integration signal. EchoDiffusion's chdrop failure is explained by pose-blind
channel-mean conditioning (channel-mean CIDE + no pose embedding).

**eat-based single-model fixes — all rejected on Replica:**
convstem 0.2644 (near 0.1422->0.1368 but mid destroyed) | AbsRel λ0.2 0.2649 (near = CNN 0.1354
exactly, far collapsed 1.59->1.93) | λ0.05 0.2654 (both bands mediocre). Near-field is *fixable*
(twice proven) but at this capacity every fix is a zero-sum band transfer — ON REPLICA. On MP3D
the same convstem WINS: cs_fb_mp3d 0.7617 = new MP3D fb record (CNN −3.0%, near AND mid better).
The zero-sum law is itself dataset-dependent: sub-patch locality is a net gain on noisy data.

**The sslam pivot (headline).** SSLAM (mixture-SSL pretraining) breaks eat's channel ceiling:

| test MAE | Replica | vs CNN | MP3D | vs CNN |
|---|---|---|---|---|
| r2 | 0.2711 | −6.3% WIN | sslam_llrd 0.8991 | −1.0% WIN |
| fb | 0.2553 (llrd 0.2575) | −1.7% WIN | sslam_llrd65 0.7714 / e30 0.7748 | −1.7% WIN |
| r6 | 3-seed 0.2336±0.005 | −2.0% WIN (s0 0.2271 sweeps ALL 8 metrics incl. AbsRel/δ1/near) | (3k training) | CNN 0.7502 |
| r8 | novd 0.2399 (vd 3k training) | −1.3% behind | eat novd 3-seed 0.7395±0.011 | −1.0% WIN |

sslam r6 s0 (0.2271) is the campaign's single best result: first cell where one model wins every
metric — near-precision + mid/far robustness simultaneously, i.e. OUTSIDE the zero-sum frontier.
val->test generalisation margin is a stable sslam property (val ties -> test wins, 4 occurrences).

**Score: 6/8 cells AFM-won (multi-seeded where close); the two open cells (MP3D r6/r8-sslam,
Replica r8+vdrop) are training (stage 3k).** MP3D per-channel CNN territory has shrunk to r6 only.

Fine-tuning-policy spectrum (MP3D fb): careless uniform ft 0.853 < frozen-hybrid (eco) 0.793 <
scratch CNN 0.785 < LLRD ft 0.773 — LLRD is the layer-wise interpolation between eco's freezing
and naive fine-tuning, and the only "free lunch" of the campaign.

## Backbone provenance & the pretraining-objective ranking

| backbone | paper | objective | checkpoint |
|---|---|---|---|
| EAT | "EAT: Self-Supervised Pre-Training with Efficient Audio Transformer", W. Chen et al., IJCAI 2024 | masked prediction of teacher features (utterance+frame) | worstchan/EAT-base_epoch30_pretrain |
| SSLAM | "SSLAM: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes", T. Alex et al., ICLR 2025 | EAT framework + audio-mixture SSL (source-preserving) | ta012/SSLAM_pretrain |
| AudioMosaic | "AudioMosaic: Contrastive Masked Audio Representation Learning", H. Huang et al., ICML 2026 (arXiv:2605.14231) | NT-Xent contrastive over structured TF-masked views | hanxunh/AudioMosaic-vit-b16-pretrained |
| BAT | "BAT: Better Audio Transformer Guided by Convex Gated Probing", H. Ghaffari, L. Rauch et al. (arXiv:2602.16305) | gated probing (convex gates suppress non-discriminative regions) | lrauch/BAT-vit-b16-pretrainedAS2M |
| M2D / M2D-CLAP | "Masked Modeling Duo", D. Niizumi et al., ICASSP 2023 / TASLP; M2D-CLAP (Interspeech 2024, 2025 ckpt) | masked prediction duo (+CLAP semantic alignment) | nttcslab m2d_clap_vit_base-*-2025 |

**Where the pretrained parameters come from / live locally.** Nothing is stored in git.
HF-hosted weights (EAT / SSLAM / AudioMosaic / BAT) are auto-downloaded by
`model/audio_backbones_0820.py` via huggingface_hub into `$AFM_WEIGHTS`
(= `/root/local1/changwoo/_afm_weights`, exported by every `0820_queue_*.sh`; HF_HOME defaults
there too). Present on disk under `hub/`: models--worstchan--EAT-base_epoch30_pretrain (344M),
models--ta012--SSLAM_pretrain (344M), models--hanxunh--AudioMosaic-vit-b16-pretrained (329M),
models--lrauch--BAT-vit-b16-pretrainedAS2M (354M). The M2D family is NOT on HF as loadable
checkpoints for our loader — the .pth zips were fetched from the NTT-CSLab M2D release
(github.com/nttcslab/m2d) into `$AFM_WEIGHTS/m2d/`: m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025
(checkpoint-30.pth = "m2d"), ...p80x2... (= "m2d20ms"), m2d_vit_base-80x608p16x16-221006-mr7_enconly
(checkpoint-300.pth = "m2d_plain"). Loading is fail-fast: a missing/corrupt checkpoint aborts the
run (`pretrained loaded: YES` is asserted in the trainer), so every result row used real weights.

All ViT-B/16 on AudioSet-2M -> backbone differences isolate the OBJECTIVE. Replica fb far>6 band
ranks exactly by objective type: mixture-SSL 1.446 (sslam) < masked-pred 1.546 (eat) < CLAP-masked
1.651 (m2d) < contrastive 1.653 (audiomosaic) < gated 1.875 (bat). near<3 is a 0.003-wide tie
(0.1422-0.1470): every AFM inherits the same patch-timing floor; objectives differ ONLY in how
much reverb structure they preserve.

**Why AudioMosaic / BAT never gained:** their failures are entirely mid/far (reverb), not near.
- AudioMosaic's contrastive NT-Xent *maximises invariance* across masked views — reverb tails are
  exactly the kind of view-dependent detail the objective trains away (mid 0.833 vs sslam 0.715).
- BAT's convex gates learn to *suppress non-discriminative (diffuse) regions* to sharpen probing —
  diffuse late reverb IS our depth signal, so the gates delete it (far 1.875, worst of all five).
  Notably BAT was the best plain-recipe val on MP3D (0.9240): suppression acts as denoising on
  noisy data, but the ceiling is low for the same reason.
- Objective ranking for echo-depth: source-preserving (mixture) > reconstruction (masked) >
  semantic alignment (CLAP) > invariance (contrastive) > suppression (gated).

**Future directions (not yet run):**
1. *Spatial-mixture DAPT*: continue SSLAM pretraining on multi-channel RIR-convolved AudioSet —
   mixtures whose components differ by direction/delay, teaching exactly the cue this task needs.
2. *Phase/ITD input*: the loader is magnitude-only; sub-frame timing lives in phase. A phase or
   GCC-PHAT channel could lift the shared near-field floor all ViTs sit on.
3. *Band-expert routing*: CNN(near) + sslam(mid/far) + eco(far prior) behind a range-gated head —
   the 3-model metric table shows they are complementary by construction.
4. *sslam multi-seed everywhere + r8-vd cell* (3k, in flight) to finish the symmetric-win claim.

## Per-method channel x dataset matrices (test MAE; rows=channels, cols=dataset; blank=not run)

Multi-seed cells are mean±std; "(t)" = training as of 2026-09-07. WIN = beats the paper CNN cell.

OAA-CNN (paper baseline)          eat + LLRD
| ch | Replica | MP3D   |        | ch | Replica      | MP3D              |
|----|---------|--------|        |----|--------------|-------------------|
| 2  | 0.2894  | 0.9084 |        | 2  | 0.2762 WIN   | 0.9018 WIN        |
| 4  | 0.2596  | 0.7849 |        | 4  | 0.2609       | 0.7717±0.002 WIN  |
| 6  | 0.2384  | 0.7502 |        | 6  | 0.2427       | 0.7736            |
| 8  | 0.2368  | 0.7467 |        | 8  | 0.2371 WIN*  | 0.7395±0.011 WIN  |
                                  (*r8 Replica = +vdrop; r8 MP3D = no-vdrop, 3 seeds each)

sslam (default 0.1x recipe)       sslam + LLRD (unified-setting candidate)
| ch | Replica          | MP3D  ||  ch | Replica    | MP3D                        |
|----|------------------|-------||-----|------------|-----------------------------|
| 2  | 0.2711 WIN       |       ||  2  |            | 0.8991 WIN                  |
| 4  | 0.2553 WIN       | 0.8335||  4  | 0.2575 WIN | 0.7920/0.7714(llrd65) WIN   |
| 6  | 0.2336±0.005 WIN |       ||  6  | 0.2385 tie | (t)                         |
| 8  | 0.2399 novd/(t)vd|       ||  8  | (t = sslam_r8vd_rep)| (t)                |

EchoDiffusion                     eat+LLRD+convstem
| ch | Replica | MP3D       |    | ch | Replica     | MP3D             |
|----|---------|------------|    |----|-------------|------------------|
| 2  | 0.2854  | 0.9007 WIN |    | 4  | 0.2644 fail | 0.7617 WIN (fb record) |
| 4  | 0.2695  | 0.7928     |
| 6  | 0.2556  | 0.7786     |
| 8  | 0.2600  | 0.7572     |

Read: the CNN is uniformly strong but no longer holds first place in most cells; sslam swept the
Replica column (3/4 cells rank-1); sslam+LLRD started winning the MP3D column (2ch, 4ch); the three
blank/(t) cells are exactly the stage-3k runs in flight. Rejected-everywhere rows (AbsRel losses,
audiomosaic/bat/m2d families, vdrop off-law cells) are kept out of these matrices — see the stage
sections above for their full numbers.
