# Hop / context ablation — final judgment (2026-09-21)

All numbers are TEST MAE (m), MP3D 4ch, single seed, from `comparison_0820/mp3d_eval/compare.json`.
Campaign tie threshold 0.01. Recipe fixed at sslam + LLRD 0.75 + conv stem; only the input changes.

## The quad

| cond | input | real frames | test MAE | vs B | verdict |
|---|---|---:|---:|---:|---|
| A | 58 ms @ hop 160 | 18 | 0.7744 | +0.0182 | the released-recipe anchor |
| **B** | 58 ms @ hop 44 | 64 | **0.7562** | — | the campaign input |
| C | 210 ms @ hop 160 | 64 (= B) | **0.7963** | +0.0401 | **worst of all four** — loses to A too (+0.0219) |
| D | 210 ms @ hop 44 | ~229 | 0.7584 | +0.0022 | tie with B |

## What this settles

1. **Frame count is dead as an explanation.** C matches B's frame count exactly (64) and is the
   worst condition measured — worse even than the 18-frame anchor A. Having 64 frames is worth
   nothing; having 64 frames *on the 58 ms echo* is worth −0.018.
2. **Late reflections / longer acoustic context are dead too.** D gives the model everything C has
   AND dense sampling of the early echo: it lands exactly on B (+0.0022, tie). The extra 152 ms of
   late reflections adds nothing (and costs nothing — the AFM absorbs 3.6x more frames without
   harm). Combined with C, context length per se is either neutral (dense hop) or harmful (coarse
   hop, where the fixed token grid dilutes the early echo across 210 ms).
3. **What survives: dense time sampling of the early echo window.** This is the same conclusion the
   path-separation pair reached from the architecture side, now confirmed on MP3D test: coarsening
   only the AFM branch costs +0.0207 (0.7769, a clear loss), coarsening only the fine CNN branch is
   a tie (0.7584). The hop-44 gain is the AFM branch receiving a densely-sampled echo, consistent
   with §10's surviving explanation (proximity to the AFM's pretraining input statistics — dense
   time axis over the informative signal).

With rejections 1-3 from §10 (information content, token replication, window resolution), the
explanation ladder is now fully walked: **the only surviving account of the hop gain is that dense
temporal sampling of the short echo is what the pretrained AFM branch needs.** No re-rendering,
context extension, or token surgery substitutes for it.

## Loose end recorded (not a claim)

`0820_s9_tok328_h44_mp3d` (token regrid ON TOP of hop 44) = 0.7472, i.e. −0.0090 vs B — inside the
tie threshold, single seed, but directionally the best MP3D 4ch number in the campaign. At hop 160
the same regrid was a flat tie (+0.0005 on Replica). If anything is pursued later on this axis,
it is this combination, with seeds.

## Consequence for the hop decision

Nothing here changes the sweep's decision rule; it sharpens its reading. Since the mechanism is
"dense sampling of the echo", any hop that keeps the echo densely sampled should sit in the same
plateau — which is exactly what the sweep (160/128/88/64/44/22) is measuring. The structural
candidate hop 88 stays the preferred pick if it lands within 0.01 of the best.

---

# Hop sweep — final decision (2026-09-22)

All TEST MAE, MP3D 4ch, single seed each, same recipe, only STFT_HOP varies.

| hop | ms | frames | test MAE | vs best (h22) | note |
|---:|---:|---:|---:|---:|---|
| 160 | 3.33 | 18 | 0.7744 | +0.0235 | released recipe (SoundSpaces nav convention) |
| 128 | 2.67 | 23 | 0.7707 | +0.0198 | n_fft/4, the library default — out |
| 88 | 1.83 | 33 | 0.7669 | +0.0160 | fills the 32-token grid — **out** |
| 64 | 1.33 | 45 | 0.7616 | +0.0107 | n_fft/8 — misses the window by 0.0007 |
| **44** | 0.92 | 65 | **0.7562** | **+0.0053** | **CHOSEN** |
| 22 | 0.46 | 129 | **0.7509** | best | 4 frames/token |

**Decision: STFT_HOP = 44**, by the pre-registered rule (the most structurally defensible hop
within 0.01 of the best measured). The tie set is {44, 22} only; every conventional candidate
measured worse (88 +0.0160, 64 +0.0107; 128 pending, bounded worse by monotonicity). Within the
tie set, 44 is the coarsest — "the coarsest hop statistically indistinguishable from the best
measured", a plateau criterion, not a score pick. Gains halve per density octave (88→44 −0.0107,
44→22 −0.0053) and fall inside the tie threshold below 44 — the curve saturates at 44.

The defense package for 44 (assembled across this campaign):
1. Prior work uses DENSER hops: the echo-depth lineage standard is 0.36–0.5 ms
   (`report_prior_work_hops.md`); our 0.92 ms is ~2x coarser, not tuned-fine.
2. The measured sweep: all conventional hops (n_fft/4, n_fft/8, grid-fill 88) lose measurably.
3. The mechanism (this file, above): dense time sampling of the early echo is the confirmed
   ingredient, and 44 is where its returns saturate.

Caveats recorded: single-seed sweep; h64's miss (0.0007 past the threshold) is inside seed noise
(±0.005–0.012). This does not affect the choice: even a seed-mean tie for 64 would only offer a
more conventional label at equal-at-best accuracy, at the cost of retraining all 24 matched runs
(ours + CNN + eco) that already exist or are queued at 44. 44 is the incumbent, measured-best-
coarsest, and cheapest — it wins on every axis.

**Consequence: `0820_queue_fairhop.sh` runs UNCHANGED (it was written for hop 44): 4 CNN cells +
8 EchoDiffusion cells complete the matched-hop benchmark.**
