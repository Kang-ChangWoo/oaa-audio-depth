#!/usr/bin/env python3
"""E142: write budget-subset prediction npz so e114_eval.py can be run UNMODIFIED.

Step selection is a function of (scene, seq, N, seed) ONLY -- never of mode. That makes the
slice-B comparison (r8 N=k vs r2 N=k) share an identical step set, hence an identical
reference cloud inside e114_eval.py (PREREG E142 sec 2-5).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

SRC = Path("/root/storage/implementation/shared_audio/EchoRecon/outputs/pred")
SCENES = ("apartment_2", "frl_apartment_5", "office_4")


def select(n_steps: int, N: int, seed, scene: str, seq: str):
    """Return sorted step-index positions (into the npz's step axis). N <= 0 means every step."""
    if N <= 0 or N >= n_steps:
        return np.arange(n_steps), "full"
    if seed == "sp":                                     # eval_fusion.py:104 convention
        idx = np.unique(np.linspace(0, n_steps - 1, N).astype(int))
        return idx, "linspace"
    # per-(scene,seq,N,seed) independent stream, mode-independent by construction
    key = f"{scene}/{seq}/N{N}/s{seed}"
    ss = np.random.SeedSequence([10_000 + 97 * int(seed), *[ord(c) for c in key]])
    rng = np.random.default_rng(ss)
    idx = np.sort(rng.choice(n_steps, size=N, replace=False))
    return idx, "random"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["r2", "r8"])
    ap.add_argument("--nviews", type=int, required=True)
    ap.add_argument("--seed", required=True)           # int-like or "sp"
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()

    src = SRC / f"{a.mode}_post"
    cell = f"{a.mode}_N{a.nviews}_s{a.seed}"
    out = Path(a.out_root) / cell
    diag = {"cell": cell, "mode": a.mode, "nviews": a.nviews, "seed": str(a.seed),
            "src": str(src), "per_seq": {}, "skipped": []}

    for sc in SCENES:
        for pf in sorted((src / sc).glob("*.npz")):
            seq = pf.stem
            z = np.load(pf, allow_pickle=True)
            steps = np.asarray(z["steps"])
            T = len(steps)
            if a.nviews > 0 and T < a.nviews:
                # PREREG 2-4: a cell only contains sequences that can supply N positions.
                diag["skipped"].append(f"{sc}/{seq}(T={T})")
                continue
            idx, how = select(T, a.nviews, a.seed, sc, seq)
            sel_steps = steps[idx]
            d = out / sc
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{seq}.npz",
                                pred=np.asarray(z["pred"])[idx],
                                steps=sel_steps)
            diag["per_seq"][f"{sc}/{seq}"] = {
                "n_steps_total": int(T), "n_used": int(len(idx)), "how": how,
                "steps": sel_steps.tolist(),
            }
    diag["n_sequences"] = len(diag["per_seq"])
    diag["n_skipped"] = len(diag["skipped"])
    Path(a.out_root).mkdir(parents=True, exist_ok=True)
    (Path(a.out_root) / f"{cell}.select.json").write_text(json.dumps(diag, indent=1))
    print(f"{cell}: {diag['n_sequences']} seq written, {diag['n_skipped']} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
