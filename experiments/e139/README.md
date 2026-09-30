# E139 — ray-mic cross-attention distance bandwidth

Launched 2026-09-30 on n4 (GPUs 4/5/6), 3 seeds × 40 epochs, E144 `ctrl` recipe otherwise.
Pre-registration: `PREREG_E139.md` (written before any number, including the mid-range rule).

## Why the 09-25 one-liner could not be run as written

The plan was "add one distance input to `RayMicAttn.bias_mlp = nn.Linear(5,64)`". Reading
`model/oaa.py` showed there is no distance value anywhere to put in:

- the five bias inputs are all directional — `[R_i^T r_j (3), ray·ear_axis (1), ear_sign (1)]`;
- the query is `self.q + dir_mlp(dir6)`, a scene-independent direction embedding;
- `self.ray_mic` is one module called **exactly once** in `forward` (`rounds` applies to
  `intra`/`inter` on the observation side), so there is no second pass that could consume a
  round-1 depth estimate;
- `aux_head` is documented as checkpoint-compatibility only and sits after the cross-attention.

So the distance axis has to live on the **query**. E139 expands each ray query into K range bins,
feeds the normalised bin centre as the 6th bias input, and mean-pools the K axis away inside
`RayMicAttn` so the decoder, ERP blocks, head, loss, data and recipe are all untouched.

## Reproducing

```
python e139_patch.py          # builds oaa_e139.py from the repo model/oaa.py (4 exact-match edits)
cp oaa_e139.py model/oaa.py   # e139_code/model shadows the repo package via PYTHONPATH order
EPOCHS=40 E139_K=4 bash run_worker_e139.sh <gpu> distq_s<seed>:none:none:<seed>
bash gate_e139.sh             # G0 gate on ep00, then fans out seeds 1-2 and watches for DONE
```

`e139_patch.py` aborts unless the source md5 is `8addeb9d193d6eef7294925fd9e67c6d` and each of the
four edits matches exactly once, so the diff cannot silently widen.

## Gates

- **G0** ep00 `val_MAE` < 3× the E144 `ctrl` ep00 baseline (2.4464 m) → 7.3392 m.
- **G2** parameter delta must be exactly **+1,088** (`dist_emb` 4×256 + `bias_mlp` first layer +64).
  Measured 2026-09-30: 15,353,642 → 15,354,730. ✅
- Verdict thresholds and the mid-range (P2) rule are in `PREREG_E139.md` §4–5. No new experiment is
  allowed to resolve the mid-range; it is decided from the distance-binned `>4 m` F1 alone.

Outputs live on the node at `/root/local1/changwoo/e139/` (not committed).
