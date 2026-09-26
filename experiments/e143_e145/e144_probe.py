#!/usr/bin/env python3
"""E144 pre-flight probe: what a 40 -> 80 epoch change actually does to the recipe.

Reads the nine existing 40-epoch runs (ctrl / bin4 / invfreq x seeds 0,1,2), reports
  * best epoch, best val MAE, and whether the curve was still descending at the end
  * measured wall-clock per epoch, to price an 80-epoch re-run
  * the LR multiplier each epoch would receive under a 40-epoch vs an 80-epoch cosine,
    which is the thing the brief asked to be checked before launching anything.
"""
from __future__ import annotations
import json, math
from pathlib import Path

RUNS = [("ctrl", f"/root/local1/changwoo/e114b/out/ctrl_s{s}", s) for s in (0, 1, 2)] \
     + [("bin4", f"/root/local1/changwoo/e135/out/bin4_s{s}", s) for s in (0, 1, 2)] \
     + [("invfreq", f"/root/local1/changwoo/e114b/out/invfreq_s{s}", s) for s in (0, 1, 2)]


def lr_factor(ep, total_ep, warmup_ep=4.0, spe=1000):
    """Reproduces train_oaa_e143.py:193-195 at the first optimizer step of epoch `ep`."""
    s = int(ep * spe)
    warm = max(1, int(warmup_ep * spe))
    total = int(total_ep * spe)
    if s < warm:
        return (s + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))


print("=" * 108)
print("THE NINE EXISTING 40-EPOCH RUNS")
print("=" * 108)
print(f"{'arm':<9}{'seed':>5}{'best_ep':>9}{'best_mae':>10}{'mae@39':>9}{'mae@30':>9}"
      f"{'last5 still improving?':>24}{'min/epoch':>11}{'epochs':>8}")
times = []
for arm, d, s in RUNS:
    p = Path(d) / "train_done.json"
    if not p.exists():
        print(f"{arm:<9}{s:>5}   MISSING {d}")
        continue
    j = json.loads(p.read_text())
    hist = j.get("hist", [])
    mae = [h.get("val_mae_m", h.get("val_mae")) for h in hist]
    mae = [x for x in mae if x is not None]
    best_ep = int(min(range(len(mae)), key=lambda i: mae[i]))
    # "still improving": is the best of the final 5 epochs better than the best of everything before?
    tail_improve = min(mae[-5:]) < min(mae[:-5]) if len(mae) > 5 else None
    secs = [h.get("sec", h.get("time_s", h.get("epoch_s"))) for h in hist]
    secs = [x for x in secs if x]
    mpe = (sum(secs) / len(secs) / 60) if secs else float("nan")
    if secs:
        times.append(mpe)
    args = j.get("args", {})
    print(f"{arm:<9}{s:>5}{best_ep:>9}{min(mae):>10.4f}{mae[-1]:>9.4f}"
          f"{(mae[30] if len(mae) > 30 else float('nan')):>9.4f}"
          f"{str(tail_improve):>24}{mpe:>11.2f}{args.get('epochs', '?'):>8}")

print()
print("=" * 108)
print("LR MULTIPLIER (fraction of peak lr) -- train_oaa_e143.py:193-195, warmup_ep 4")
print("  total = a.epochs * steps_per_ep, so the cosine PERIOD is tied to --epochs.")
print("=" * 108)
print(f"{'epoch':>7}{'40-epoch run':>15}{'80-epoch run':>15}{'ratio 80/40':>14}")
for ep in (0, 4, 8, 16, 24, 29, 31, 33, 35, 39, 40, 48, 60, 72, 79):
    a = lr_factor(ep, 40)
    b = lr_factor(ep, 80)
    r = (b / a) if a > 1e-6 else float("inf")
    tag = "   <- 40ep budget ends here" if ep == 39 else ""
    print(f"{ep:>7}{a:>15.4f}{b:>15.4f}" + (f"{r:>14.2f}" if r != float("inf") else f"{'inf':>14}") + tag)

print()
if times:
    mpe = sum(times) / len(times)
    print(f"measured mean wall-clock per epoch over the runs that logged it: {mpe:.2f} min")
    print(f"  40 extra epochs x 9 runs = {40*9*mpe/60:.1f} GPU-hours")
    print(f"  80 epochs from scratch x 9 runs = {80*9*mpe/60:.1f} GPU-hours")
else:
    print("no per-epoch timing field found in train_done.json -- price from logs instead")
