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
