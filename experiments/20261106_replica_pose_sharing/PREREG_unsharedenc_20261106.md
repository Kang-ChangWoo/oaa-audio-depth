# PREREG - weight-sharing ablation across observations (unshared-enc), 2026-11-06

## Question
The reference model encodes every (yaw, ear) observation with ONE weight-shared encoder. Is that
sharing load-bearing, or is it just a parameter saving? Changwoo's question (2026-11-06): keep
everything else and give each observation its OWN encoder of the same architecture.
No row in the paper answers this today: the only non-shared comparators are ResNet (.282) and
BatVision (.205), i.e. different backbones. This is the matched-backbone version.

## Arms (Replica, 8 observations, hop 160, 40 epochs, bs 3 x accum 11, warmup 8, subset-aug)
- control: `paperrep_ctrl_r8_s0..s4`, LANDED, test AbsRel .12903 .13172 .14137 .13569 .20713
  -> mean .14899, sd .03283 (n=5)
- unshared: `paperrep_unshared_r8_s0..s4`, 5 seeds, byte-identical recipe + `--unshared-enc`

Population fixed at 5 v 5 BEFORE any number is read; no conditional second batch of seeds.

## What the switch does (verified, not asserted)
Same `ViewEncoder` architecture, instantiated `nviews` times; observation i is encoded by encoder i.
Observation 0 keeps the `enc.*` state-dict keys, the other seven live under `enc_extra.*` and are
only created when the switch is on. Untouched: input tensor, token grid (4,096 acoustic tokens),
the 8 pose codes, intra/inter-observation attention, the ray path, the decoder, the recipe.

Smoke (2026-11-06, commit below):
- switch OFF is bit-identical to the pre-patch tree: 18 cases (nviews/in_ch = 8/1, 8/2, 8/3, 8/4,
  8/8, 1/8, 4/2, 2/1, each with and without a dropped observation, plus two at the released
  256x512 encoder resolution) -> sha256 of the output map equal in all 18, mismatches 0
- switch ON: perturbing only encoder 3 moves observation 3's tokens (max |d| 331.6) and moves the
  other seven by EXACTLY 0.0; the shared path moves all eight. The sharing is really gone.
- params: encoder 6,394,496 (41.6 % of the model). Total 15,353,642 -> 60,115,114, i.e. x3.92,
  and the identity total_unshared = total_shared + 7 x encoder holds exactly.
- a reference checkpoint still loads strict=True with the switch off; an unshared checkpoint
  round-trips strict=True with it on; `core/ckpt.py` forwards the flag, so scoring and
  `--resume auto` rebuild the same architecture.
- one backward + AdamW step runs; the only parameters without a gradient are `aux_head.*`, which
  is the same in the control (kept for checkpoint compatibility, not used by the loss).
- VRAM: +716 MB over the control from params/grads/Adam states alone ([inference]: 7 x 6.39 M x
  16 B). Control sits at ~18.3 GB of a 24 GB 3090, so ~19.0 GB expected. Activations are
  unchanged (same observations, same shapes). To be CONFIRMED from the first epoch, not assumed.

## Primary metric
test AbsRel, per-image cos-latitude weighted (`core.evaluate`), best.pth. Pre-registered, not
revisable after the fact.

## Decision rule (registered BEFORE the runs)
- unshared mean WORSE than control, Welch two-sided p < 0.05 -> PASS: the weight sharing is
  load-bearing and the paper gets the row, with the capacity caveat below stated in the same
  sentence.
- unshared mean EQUAL OR BETTER -> sharing is not carrying its weight at this data scale. Report
  immediately; the paper then needs the matched-capacity control before any claim.
- neither (p >= 0.05) -> conclusion is "no detectable difference at 5 v 5 against a control sd of
  22 % of the mean". The population does NOT get extended to chase significance.

## Prediction (registered)
Unshared lands WORSE than control by more than 5 %, i.e. AbsRel >= .156, and NOT catastrophically
so (clearly better than stackall's predicted .22) [inference]. Basis: Replica trains on 12 scenes
while the parameter count goes x3.92, and the eight observations are near-identical statistically
(the heading-signature probe found the direct-sound ILD/ITD bit-identical across datasets), so
sharing is a good prior rather than a constraint. Nothing is removed from the architecture here,
which is why the damage should be far smaller than removing the posed tokens.

## Confounds and adverse signals to report either way
- CAPACITY, the main one: a reviewer will say the loss comes from 3.92x the parameters on 12
  scenes, not from unsharing. This arm alone cannot separate the two. The control that can is a
  matched-capacity pair: either widen the SHARED encoder to ~60 M params, or shrink each
  per-observation encoder so the total matches 15.35 M. Not queued yet; decision pending.
- Replica 8 obs has the widest seed floor measured (control sd 22 % of the mean) and the
  subset-aug stream is unseeded, so 5 v 5 resolves only large effects.
- Every control run peaked at ep 36-39, so the 40-epoch budget is not a converged comparison, and
  a 3.92x model may need more epochs than a 1x model to converge. If the unshared arm's best
  epoch is 39 in most seeds, the comparison is budget-limited and that must be said.
- Optimizer state, not just params, scales: the effective batch (3 x 11 = 33) and LR are held
  byte-identical to the control, which is the fair choice but not a tuned one for a larger model.

---

# EXTENSION 2026-11-06 19:5x KST - Replica is the primary dataset for this ablation

Changwoo: "이 실험들은 주로 replica에서 해야하는거야" (and the same for the random-heading
ablation). So the weight-sharing ablation runs on ALL THREE Replica rungs, each against the
control population that actually exists, registered before any of them is read:

| rung | arm cells | control (landed) | control mean | params unshared | epoch |
|---|---|---|---|---|---|
| 2 obs (r2) | `paperrep_unshared_r2_s0..s4` | `paperrep_ctrl_r2_s0..s4` (5) | .16102, sd .00403 | 21,748,138 (x1.416) | 180 s -> 2.0 h |
| 4 obs (fb) | `paperrep_unshared_fb_s1..s4` | `paperrep_ctrl_fb_s1..s4` (4) | .14636 | 34,537,130 (x2.249) | measured at launch |
| 8 obs (r8) | `paperrep_unshared_r8_s0..s4` | `paperrep_ctrl_r8_s0..s4` (5) | .14899, sd .03283 | 60,115,114 (x3.915) | 749.6 s -> 8.3 h |

Encoder = 6,394,496 params in every rung (in_ch 1), so the extra cost is exactly
(nviews - 1) x 6,394,496 and the capacity confound GROWS with the observation count.

Queue placement: r2 on n14 (lines 1264-1268, inserted AHEAD of the `gibh160_itd_r2_s1..s4`
cells on Changwoo's Replica-first instruction, and AFTER the `mp3dh160n14_*_r2_s3/s4` cells so
that already-registered 5v5 population is not left half-finished); fb on n1 (lines 748-751);
r8 on n1 (lines 740-744, s0/s1 already training).

## Reading rule for the three rungs (registered)
- Each rung is judged on its OWN bar: worse than its control with Welch two-sided p < 0.05 at the
  fixed population above. No rung may borrow significance from another.
- The 2-obs rung is the CLEANEST test of weight sharing, because unsharing costs only +41.6 %
  parameters there against +291.5 % at 8 obs. If sharing is load-bearing, the 2-obs rung should
  already lose.
- The across-rung trend (does the damage grow with observation count?) is DESCRIPTIVE, not a
  test. A monotone trend is equally consistent with "sharing matters more with more views" and
  with "the capacity penalty grows", and only the matched-capacity control can separate those.
- Prediction (registered): 2 obs loses by less than 8 obs; the 2-obs arm is where I am least
  confident, and a 2-obs PASS is the strongest single result this ablation can produce
  [inference].

## Control correction (before any arm number existed), 2026-11-06 20:0x
4-obs control is paperrep_ctrl_fb_s1..s4 = .143603 .149621 .149018 .144258 -> mean .146625
(results/repseeds_n1_20261102.json). The .14636 in the table above was a rounded note; the
judgement uses .146625 and the four seed values, which the pose-ablation run just reproduced
bit for bit in its true mode (.14360 for s1), confirming the scoring path.

## Capacity-matched control: DROPPED by Changwoo, 2026-11-06 20:20 KST
"용량 맞춘 대조군은 딱히 필요 없고". So the x1.42 / x2.25 / x3.92 parameter increase stays as a
STATED LIMITATION of this ablation and is not separated experimentally. When the row is written,
the capacity confound must be named in the same sentence as the result, and the 2-observation
rung (+41.6 % params only) is the one to lead with for exactly that reason.
