# PREREG - channel-stacking tokenisation ablation (stackall), 2026-11-06

## Question
Does the architecture need ONE POSED TOKEN PER OBSERVATION, or would stacking every channel
into a single encoder input do as well? The reviewer question is "why not just stack the
channels", and today the only answer in the paper is ResNet (.282) / BatVision (.205) on Gibson
2 obs -- baselines with a DIFFERENT BACKBONE, so they cannot isolate the tokenisation.

## Arms (Replica, 8 observations, hop 160, 40 epochs, bs 3 x accum 11, warmup 8, subset-aug)
- control: `paperrep_ctrl_r8_s0..s4`, LANDED, test AbsRel .12903 .13172 .14137 .13569 .20713
  -> mean .14899, sd .03283 (n=5)
- stackall: `paperrep_stack_r8_s0..s2`, 3 seeds, byte-identical recipe + `--pair-feat stackall`

`stackall` does NOT touch the input tensor (build() is the identity, verified bit-exact in the
smoke). The model is built with nviews=1 and in_ch=8, so the same bytes are read as one stacked
token. What is removed is the per-view pose/ray embedding and the cross-view attention. The four
headings of the 8-channel rig collapse into one token whose pose is the first channel's (0 deg, L)
-- a pose IS invented, which is the thing under test. Params 15.355658 M vs 15.353642 M (+2,016).

## Primary metric
test AbsRel, per-image cos-latitude weighted (`core.evaluate`), best.pth. Pre-registered, not
revisable after the fact.

## Decision rule (registered BEFORE the runs)
- stackall mean WORSE than control and Welch two-sided p < 0.05 -> PASS: posed per-view tokens
  are necessary, and this is the ablation row that answers the reviewer.
- stackall mean worse but p >= 0.05 -> add seeds up to 5 v 5 before concluding anything.
- stackall mean EQUAL OR BETTER -> the per-view tokenisation is not carrying its weight. That is
  a bigger finding than the pass case and gets reported immediately, not buried.

## Prediction (registered)
stackall lands >= 50 % worse than control, i.e. AbsRel >= .22 [inference]. Basis: folding only
HALF the views (stereoild, 8->4) already moved Replica from .14899 to .38677 (+160 %), and the
inference-only pose ablation showed heading corruption costs +272~430 % AbsRel.

## Adverse signals to report either way
- Replica 8 obs has the widest seed floor we have measured (control sd 22 % of the mean) and the
  subset-aug stream is unseeded, so 3 v 5 only resolves large effects.
- Every control run peaked at ep 36-39, i.e. the 40-epoch budget is not a converged comparison.
