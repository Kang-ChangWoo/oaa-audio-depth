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

Multi-seed cells are mean±std; "(t)" = training as of 2026-09-08. Strict rule: WIN = mean gap >=0.01m vs the paper CNN cell; anything under 0.01 is a tie ("tie(+)" = AFM-side direction).

OAA-CNN (paper baseline)          eat + LLRD
| ch | Replica | MP3D   |        | ch | Replica      | MP3D              |
|----|---------|--------|        |----|--------------|-------------------|
| 2  | 0.2894  | 0.9084 |        | 2  | 0.2762 WIN   | 0.9018 tie(+)     |
| 4  | 0.2596  | 0.7849 |        | 4  | 0.2609       | 0.7717±0.002 WIN  |
| 6  | 0.2384  | 0.7502 |        | 6  | 0.2427       | 0.7736            |
| 8  | 0.2368  | 0.7467 |        | 8  | 0.2371 tie*  | 0.7395±0.011 tie(+) |
                                  (*r8 Replica = +vdrop; r8 MP3D = no-vdrop, 3 seeds each)

sslam (default 0.1x recipe)       sslam + LLRD (unified-setting candidate)
| ch | Replica          | MP3D  ||  ch | Replica    | MP3D                        |
|----|------------------|-------||-----|------------|-----------------------------|
| 2  | 0.2711 WIN       | 0.8965 WIN | 2 | 0.2805 tie(+) | 0.8991 tie(+)          |
| 4  | 0.2659±.012 tie  | 0.8335||  4  | 0.2575 tie(+) | 0.7803±.009 tie(+) (llrd65 2s, s1 arbiter queued) |
| 6  | 0.2336±0.005 tie(+) | (t)||  6  | 0.2385 tie | 0.7848 LOSS                 |
| 8  | 0.2399 novd/0.2368 vd tie ||  8 | 0.2363 vd tie(best r8) | 0.9861 dead-run (s1 retry) |

Beyond-I2D (audio-only port, 318M; Parida et al. CVPR'21)
| ch | Replica | MP3D |
|----|---------|------|
| 2  | 0.3150 LOSS (vs CNN -0.026) | (t) |
| 6  | 0.2982 LOSS (vs CNN -0.060) | (t) |
| 4  | 0.3125 LOSS (vs CNN -0.053; converged, val 0.330) | (t) |
(r6/r8 rep + all mp3d: training/queued)

EchoDiffusion                     eat+LLRD+convstem
| ch | Replica | MP3D       |    | ch | Replica     | MP3D             |
|----|---------|------------|    |----|-------------|------------------|
| 2  | 0.2854  | 0.9007 tie(+) | | 2  | 0.2754 WIN  | 0.8884 WIN (r2 record) |
| 4  | 0.2644 fail | 0.7617 WIN (fb record) |
| 4  | 0.2695  | 0.7928     |
| 6  | 0.2556  | 0.7786     |
| 8  | 0.2600  | 0.7572     |

Read: the CNN is uniformly strong but no longer holds first place in most cells; sslam swept the
Replica column (3/4 cells rank-1); sslam+LLRD started winning the MP3D column (2ch, 4ch); the three
blank/(t) cells are exactly the stage-3k runs in flight. Rejected-everywhere rows (AbsRel losses,
audiomosaic/bat/m2d families, vdrop off-law cells) are kept out of these matrices — see the stage
sections above for their full numbers.

## Mechanism experiment: late-reverb / early-arrival ablation (test-time, no training)

`0820_reverb_ablation.py` masks the test waveform inside data_0422's STFT (spec cache disabled;
clean condition reproduces compare.json to 4 decimals) and re-evaluates the SAME frozen fb models.
Full numbers: comparison_0820/reverb_ablation.json.

dMAE vs clean (Replica fb test):
| perturbation | CNN | eat_llrd | sslam |
|---|---|---|---|
| earlyzero1m (delete direct/early, keep tail) | +0.706 (near +0.431) | +0.459 | +0.293 (mid only +0.67) |
| latecut6m (delete tail, keep <6m primaries)  | +0.146 | +0.094 | +0.113 (dmid MAX +0.618) |
| latecut4m | +0.325 | +0.254 | +0.271 |

Three verdicts: (1) the CNN binds depth to local early arrival-time cues — deleting them breaks it
2.4x harder than sslam; (2) sslam's mid-band advantage lives in the late tail — with mid-surface
primaries preserved but the tail cut, its clean-time mid edge (−0.043) vanishes and inverts, the
largest dmid of the three models; (3) the naive total-dMAE prediction (dSSLAM >> dCNN under latecut)
is refuted in an informative way: sslam degrades LEAST under every deletion (far d +0.556 vs CNN
+0.782 even under latecut) — its representation is *distributed*, reconstructing from whatever
component survives, while the CNN is the most fragile under any distribution shift (even its near
band doubles sslam's dnear under latecut). TERMINOLOGY CORRECTION (important): the input window is already truncated at 2799 samples =
10 m round-trip (58 ms), so NO model ever sees RT60-style late diffuse reverberation. What latecut
removes is the *trailing portion of the truncated echo train*: far-surface primaries + low-order
multi-bounce components with total path 12-20 m (which do land inside the window). Mechanism
sentence, corrected: mixture-SSL transfers to echo-geometry because it encodes depth in the
*distributed multi-path structure within the 58 ms echo train* (overlapping discrete arrivals —
exactly what mixture pretraining teaches it to disentangle), not in diffuse reverb statistics;
the task CNN binds depth to local early arrival-time cues. This also explains the pretraining-objective far>6 ranking (mixture <
masked < contrastive < gated): objectives differ in how much of that distributed structure they
preserve.

## Mechanism campaign, round 2 (2026-09-07): controls and refinements

Discussion-driven corrections and the experiments they spawned (all test-time, frozen models,
Replica fb; scripts: 0820_reverb_ablation.py {v1|sweep|notch|renorm|seeds|shuffle}):

1. **What "direct sound" is here.** The rig self-emits; waveforms are rendered IRs. The direct
   spike sits at a FIXED 49 samples (1.02 ms — emitter-to-ear rig geometry) in every scene,
   ~1.5 ms wide, 44-72% of window energy. It carries zero scene information; first reflections
   (scene-dependent, >=280 smp for >=1 m surfaces) do not overlap it. STFT smearing feathers any
   time cut by ~±0.7 m equivalent (symmetric across models).
2. **"Big-signal / normalisation" hypothesis — REFUTED by the renorm control.** Zeroing the onset
   and then restoring waveform energy made every model WORSE (cnn dMAE +0.706->+0.810,
   eat +0.459->+0.535, sslam +0.293->+0.324). The AFMs self-normalise per sample anyway (log1p+std),
   so the statistics shock was never the driver; boosting surviving echoes distorts the
   amplitude-distance cue instead. The earlyzero damage is therefore genuine loss of onset-region
   features — and the CNN (which has NO input normalisation) losing most confirms its readout is
   onset-bound (arrival-time + absolute-amplitude reference).
3. **Latecut sweep (3/5/7/8/9 m): the advantage is DISTRIBUTED, and it grows with distance.**
   Clean sslam-vs-CNN edge by band: near +0.007 (CNN), mid -0.043, far -0.074 — the AFM edge is
   proportional to distance (weaker/ambiguous primaries -> multi-bounce corroboration matters more).
   Cutting only the final 8-10 m segment already flips the mid edge (-0.043 -> +0.056), and each
   earlier cut deepens it smoothly (7 m +0.079, 5 m +0.168, 3 m +0.212): no cliff at any single
   slot -> the edge is integrated thinly across the whole tail. One-witness (CNN onset, cliff
   collapse under earlyzero) vs many-small-clues (sslam tail, graceful degradation) is exactly the
   distributed-representation signature. Caveat: absolute dMAE inflates with OOD-ness as the cut
   moves earlier; read the sslam-CNN differential, not absolute values.
4. **In flight**: notch matrix (zero one 2 m slot at a time -> which DEPTH bands break: CNN should
   be diagonal / slot-local, sslam off-diagonal); seed replication (sslam s1 true seed + llrd
   recipe variant + oaa_fb_vw same-arch retrain — does the earlyzero/latecut8 pattern reproduce,
   ruling out single-seed artefacts); shuffle-tail (permute 0.2 m blocks after 6 m, energy
   preserved, structure destroyed — if sslam drops like latecut6m it reads STRUCTURE, not energy).

## Mechanism campaign, round 3: notch matrix, seed replication, shuffle-tail — final story

**Notch damage matrix** (zero one 2 m depth-slot; dMAE per band): CNN is slot-local (diagonal):
its far band loses +0.123 when the far slot is cut, mid mostly from its own slot. sslam is
cross-slot: mid loses ~2x the CNN amount when 6-8 m or 8-10 m slots are cut (mid primaries
untouched), and — the standout cell — sslam's far band loses only +0.019 when its OWN primary
slot (8-10 m) is removed (CNN +0.123, eat +0.111): sslam reconstructs far depth largely without
far primaries.

**Seed replication** (independent trainings: oaa_fb_vw retrain, sslam s1 true seed, sslam_llrd):
every direction reproduces — earlyzero collapse CNN 1.55x sslam (orig 2.4x); latecut8 tail
dependence AFM 4-7x CNN. Magnitudes fluctuate with seed (sslam mid d at latecut8: 0.195 -> 0.040):
qualitative claims are seed-robust, single-seed magnitudes should not be quoted as precise.

**Shuffle-tail** (permute 0.2 m blocks after 6 m; energy preserved, temporal arrangement destroyed):
harmless to every model (dMAE +0.003..+0.008 vs latecut6m +0.09..+0.15). Combined with the notch
matrix (damage ~ proportional to removed energy, similar for 6-8 vs 8-10 slots), this REFINES the
mechanism: what the models — sslam best of all — read from the tail is its *aggregate reverberant
energy/spectral statistics* (decay/DRR-like cues), not a precisely-timed arrangement of individual
multi-bounce arrivals.

**Final mechanism statement.** The task CNN binds depth to local onset features (first-arrival
timing + absolute-amplitude reference): single-witness readout, cliff collapse when the onset is
perturbed, slot-diagonal damage. SSLAM carries a distributed readout: graceful degradation under
any component deletion, cross-slot integration (far reconstructed without far primaries), and a
better extraction of aggregate tail statistics — consistent with mixture-SSL pretraining teaching
robust spectral statistics under superposition. Four independent manipulations (earlyzero+renorm,
latecut sweep, notch matrix, shuffle-tail) and a seed replication all point the same way.

## Mechanism campaign, round 4 (2026-09-07/08): the four causal-gap controls

G. **Timing vs amplitude split**: shifting the direct impulse +0.2 m (amplitude preserved) breaks
all three models similarly (d +0.59..0.74, CNN/sslam ratio 1.26) — the temporal anchor is a shared
strategy. Scaling the direct impulse to 0.25x (position preserved) breaks CNN 2.7x more than sslam
(+0.463 vs +0.175) — the model-separating dependence is AMPLITUDE calibration (echo-to-direct
ratio), not timing per se.
H. **Shuffle block-size sweep** (1 m -> 0.02 m): all deltas stay <= +0.009 at every scale.
Bag-of-patches is rejected along with fine-timing: the tail cue is aggregate energy/spectral
envelope surviving the STFT window — required temporal resolution ~ none.
I. **First-window x tail 2x2** (notch 3-6 m, 8-10 m, both): mid-band damage is SUBadditive for
sslam only (0.994 predicted vs 0.943 observed — partial redundancy/substitution between sources);
superadditive for CNN (0.167->0.224) and eat (0.819->1.093). Quantitative interaction evidence for
distributed/redundant encoding.
F+. **Seed replication of mechanism magnitudes**: sslam tail dependence is stable across
independent trainings (latecut6 d = 0.113/0.109/0.133); shuffle harmlessness reproduces everywhere
(<=0.016). CNN's latecut magnitude swings 5x across checkpoints (0.146 fin vs 0.030 vw; vw's
recipe includes view weighting so it is not a pure seed replicate) — the claim "the CNN also
depends heavily on the tail" is RETRACTED; strong CNN claims rest only on seed-reproduced
manipulations (earlyzero 1.55-2.4x, amplitude 2.7x).

Also: unified-setting Replica r2 landed — sslam_llrd_r2_rep test 0.2805 vs CNN 0.2894: gap 0.0089
< 0.01 -> TIE under the pre-registered rule (sslam default recipe 0.2711 remains a win; the
LLRD-on-Replica cost ~ +0.005..0.009 reproduces at r2 and r6).

**Replica column CLOSED (2026-09-08)**: sslam_r8vd_rep test MAE 0.2368 — equal to the paper CNN to
4 decimals (RMSE 0.4785 vs 0.4810 and far 1.435 vs 1.589 better; near/mid worse). The r8 vdrop law
transfers to sslam (novd 0.2399 -> vd 0.2368). Final Replica verdicts for the sslam family:
r2 WIN (-0.018) / fb tie (2 seeds, s2 arbiter running) / r6 WIN (3 seeds) / r8 exact tie.
sslam never loses a Replica cell.

**fb Replica arbiter (s2) landed**: 0.2789 -> 3-seed 0.2553/0.2636/0.2789, mean 0.2659±0.012 vs
CNN 0.2596. TIE under the <0.01 rule, leaning CNN; the early single-seed "sslam fb win" is
formally retired. sslam fb seed spread (0.012) is the largest measured. Final Replica column
(sslam family): r2 WIN / fb TIE / r6 WIN (3 seeds) / r8 exact TIE — never loses, clear wins at
r2 and r6.

**Unified-setting Replica column CLOSED**: sslam_llrd_r8vd_rep test 0.2363 — best r8 of the
campaign (RMSE 0.4718 / mid 0.6698 / far 1.3952 all beat the CNN; MAE gap 0.0005 = tie by rule).
The LLRD-on-Replica cost seen at r2/r6 (+0.005..0.009) vanishes at r8. Unified sslam+LLRD now:
Rep r2/fb/r6/r8 all ties (never loses), MP3D r2/fb wins, r6/r8 training.

### 2026-09-08: sslam_r2_mp3d lands — plain sslam wins the MP3D 2ch cell too
Test MAE 0.8965 vs CNN 0.9084 (gap 0.0119 > tie threshold 0.01 → WIN; also below eat_llrd 0.9018
and sslam_llrd 0.8991, both within-tie of each other). With this, plain sslam has never lost a
completed cell: Replica r2/fb/r6 WIN, r8 exact tie; MP3D r2 WIN. Queue note: beyond(ITD) stage-3n
now has queue priority per user; cs_r6/r8_rep deferred via placeholder until all 8 beyond jobs
dispatch (stage3o auto-requeues them).

### 2026-09-08: strict tie-rule audit (user-prompted relabel)
Applying the <0.01m tie rule uniformly relabels several cells previously called WIN:
eat MP3D r2 (gap .0066), eat MP3D r8 (.0072), sslam Rep r6 (.0048), sslam_llrd MP3D r2 (.0093),
sslam_llrd Rep fb (.0021), EchoDiff MP3D r2 (.0077) are all ties (direction noted). Strict wins
that survive: sslam Rep r2 (.0183), sslam MP3D r2 (.0119), eat Rep r2 (.0132), eat/llrd65 MP3D fb
(.0132/.0135), cs MP3D fb (.0232). eat MP3D r6 is a strict LOSS (.0234). Honest headline:
sslam family never loses a completed cell; strict wins concentrate at sparse-channel (r2) and
hard-data (MP3D fb) cells. Report §3/§8 and the matrix updated to match.

### 2026-09-08 (2): cs_r2_mp3d lands — convstem takes the MP3D 2ch record
eat+LLRD+convstem MP3D r2 test 0.8884 vs CNN 0.9084 (gap 0.0200, strict WIN) — new best for the
cell, below sslam 0.8965. Full metrics: RMSE 1.3708, d1 0.4666 (cell best), near 0.4494 (beats CNN
0.4563 — the "convstem is a net gain on rough MP3D, zero-sum only on clean Replica" law reproduces
at 2ch), mid 1.2577, far 4.2623. MP3D column now: AFM strict wins at r2 (two families) and fb
(three runs); CNN holds r6 only. Freed GPU took 0820_beyond_r2_rep (ITD queue #3).

### 2026-09-09: sslam_llrd_r6_mp3d lands — MP3D r6 stays with the CNN (strict LOSS)
Unified-setting test 0.7848 vs CNN 0.7502 (gap 0.0346 > 0.01). Second AFM family to lose this
cell (eat+LLRD 0.7736). Band structure is the usual trade: far better (3.803 vs 3.940) but
near (0.400 vs 0.367) / mid (1.113 vs 1.103) / d1 (0.533 vs 0.559) all worse — with six clean
observations on messy MP3D the near/mid precision loss dominates. Verdict split: plain sslam
still unbeaten on its 5 completed cells; unified sslam+LLRD is now 1 win / 5 tie / 1 LOSS with
MP3D r8 pending. The "one setting never loses anywhere" goal fails at MP3D r6.

### 2026-09-09 (2): sslam_llrd_r8_mp3d — dead run (val plateau), s1 retry queued
Test 0.9861 vs CNN 0.7467: nominally a strict LOSS, but the val trajectory shows a collapsed
run — floor 1.1102 at ep6, then monotone worsening to 1.178 by ep29 (near-constant-depth trap +
late overfit). Same recipe trained fine at r6 (val 0.90 -> test 0.785) and eat novd r8 reached
0.7395, so this reads as seed/cell-specific instability, not a verdict on the setting. vdrop
contamination ruled out (subset_aug=False gates vdrop off; stored kmax=4 is argparse default).
stage3p queues a seed-1 retry that waits for all beyond dispatches (user priority) then takes
an empty GPU. Cell verdict deferred to the retry; if s1 also collapses, record the LOSS.

### 2026-09-09 (3): first Beyond-I2D cell — beyond_fb_rep 0.3125 (strict LOSS)
Converged cleanly (val 0.3301, d1 0.760) but lands below every main baseline: CNN 0.2596,
sslam family 0.2553-0.2659, EchoDiffusion 0.2695; only EchoScan (0.3516) is worse. First
evidence that the ITD-style multi-branch attention port is uncompetitive against
position-aware geometric attention on this task.

### 2026-09-09 (4): missed eval recovered — cs_r2_rep 0.2754, a strict WIN on clean Replica
eatllrd_cs_r2_rep (finished earlier, eval overlooked) tests at 0.2754 vs CNN 0.2894
(gap 0.0140 > 0.01). This REFINES the convstem law: "zero-sum on clean Replica" holds only
at fb (0.2644 near-for-mid trade); at r2 convstem is a net gain on BOTH datasets
(Rep 0.2754 WIN, MP3D 0.8884 record). New law: convstem helps wherever the problem is
underdetermined (sparse mics) or noisy (MP3D); it only trades on clean+well-observed cells.
cs scorecard: 3 strict wins (Rep r2, MP3D r2 record, MP3D fb record), 1 fail (Rep fb).
Also noted: 0820_sslam_r8_mp3d (plain sslam MP3D 8ch) still training ep12/30 — added to watch.

### 2026-09-09 (5): beyond_r2_rep 0.3150 — second Beyond-I2D cell, strict LOSS
vs CNN 0.2894 (gap -0.0256) and far behind every AFM (sslam 0.2711, cs 0.2754). Notably beyond
shows almost NO channel scaling on Replica (r2 0.3150 vs fb 0.3125 — only 0.0025 apart, where
every other model gains 0.02-0.03 from 2->4 mics): consistent with its position-blind channel
handling failing to exploit added observations, the same diagnosis as EchoDiffusion's
non-monotonic scaling but even flatter.

### 2026-09-10: beyond_r6_rep 0.2982 — third Beyond-I2D cell, strict LOSS
vs CNN 0.2384 (gap -0.0598). Replica scaling picture for beyond: r2 0.3150 / fb 0.3125 /
r6 0.2982 — some gain finally appears at 6 mics but the deficit vs CNN WIDENS with channels
(-0.026 -> -0.053 -> -0.060): every extra observation helps the position-aware models more
than it helps beyond. Only beyond_r8_rep remains on Replica.

### 2026-09-10 (2): llrd65_fb_mp3d_s2 lands at 0.7892 — the llrd65 MP3D fb "win" falls to a tie
Seed mean now 0.7803+-0.009 vs CNN 0.7849: gap 0.0046 < 0.01 -> tie(+). The single-seed 0.7714
win was seed luck, the exact same failure mode the fb Replica cell taught us (0.2553 -> 3-seed
tie). Arbiter s1 was found DEAD AT STARTUP (CUDA OOM when a co-tenant grabbed the GPU at
dispatch time) — log archived as .oomcrash, stage3q requeues it on an empty GPU. The MP3D fb
cell itself KEEPS its strict wins independently: eat+LLRD 3-seed 0.7717+-0.002 (gap 0.0132)
and cs 0.7617 (gap 0.0232). Consequence for the unified-setting scorecard: its only win came
from the llrd65 variant, so unified sslam+LLRD is now 0 W / 6 T / 1 L (+ r8 retry pending);
the "wins under hard conditions" claim now rests on eat+LLRD and cs at MP3D 2/4ch and plain
sslam at r2 — unchanged at the cell level, reshuffled at the family level.
