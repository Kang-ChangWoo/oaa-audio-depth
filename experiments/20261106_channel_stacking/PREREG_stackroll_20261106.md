# PREREG - stackroll, the mix-and-keep hybrid, 2026-11-06 (Changwoo's proposal)

## What it is
Every observation group sees ALL eight channels, so they mix in the stem the way a stacking
encoder mixes them, but there are still 8 posed groups, 4,096 tokens and one shared-weight
encoder. Group g is the channel list rolled so channel g is first. Verified: group i plane j is
channel (j+i) mod 8 for all 64 planes, every group holds every channel exactly once, and a
--subset-aug drop removes that observation from all eight groups.

This is NOT an ablation. It is a candidate architecture that could beat the reference, because it
adds early channel mixing on top of the posed tokenisation instead of trading one for the other.

## Arms (Replica, 8 observations, hop 160, 40 epochs, bs 3 x accum 11, warmup 8, subset-aug)
- control  `paperrep_ctrl_r8_s0..s4` LANDED, test AbsRel mean .14899, sd .03283 (n=5)
- stackall `paperrep_stack_r8_s0..s2` RUNNING (1 group, 512 tokens, 1 pose)
- stackroll `paperrep_roll_r8_s0` 1 seed on the one free card
Params: ctrl 15,353,642 | stackall and stackroll 15,355,658 (+2,016, the stem's extra planes).

## Primary metric
test AbsRel, per-image cos-latitude weighted (`core.evaluate`), best.pth. Registered up front.

## Decision rule (one seed only, so this is a SCREEN, not a verdict)
- stackroll beats the control mean by more than the control's own seed spread (.033) -> promote to
  5 seeds and treat it as a model change, not an ablation row.
- stackroll inside +-1 sd of the control -> no evidence either way at n=1; decide by whether
  stackall landed worse (then posed tokens matter and mixing is free) or not.
- stackroll clearly worse than the control -> early channel mixing hurts even with poses kept,
  which strengthens the paper's tokenisation claim.

## Prediction (registered)
stackroll lands within +-1 sd of the control, i.e. no clear gain [inference]. The channels already
meet at the inter-observation attention, so mixing them earlier mostly removes the encoder's
ability to build a clean per-observation representation first.

## Adverse signals to report either way
- n=1 against a 5-seed control whose sd is 22 % of its mean: only a large effect is visible.
- Each group now carries 8x the input planes, so the stem sees a different input distribution than
  the control's single plane; the +2,016 params are not the only asymmetry.
