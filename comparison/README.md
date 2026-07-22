# comparison/ — baseline results

All comparison-baseline training outputs and the aggregated eval table live here. "Ours"
(OAA / BatVision) trains to `../out/`; the baselines below train to this folder.

## Layout

```
comparison/
  <run-name>/best.pth   <run-name>/last.pth   <run-name>/train_done.json   # one dir per run
  compare.json                                                             # eval summary table
```

## Train a baseline (writes here by default)

```bash
python3 train_baseline.py --model resnet   --run-name rn_r8   --mode r8
python3 train_baseline.py --model vit      --run-name vit_r8  --mode r8
python3 train_baseline.py --model beyond   --run-name byd_cB  --mode cB
python3 train_baseline.py --model echoscan --run-name es_s0
# EchoDiffusion trains from its isolated env (see ../model/echodiffusion.py); point its
# --out-dir here too so all baseline ckpts stay together.
```
Each run prints a per-module parameter breakdown at startup (`[params] ...`).

## Evaluate (ours + baselines in one table)

```bash
# eval.py searches out/ (ours) then comparison/ (baselines); summary -> comparison/compare.json
python3 eval.py --run-name oaa_r8_s0 bat_r8_s0 rn_r8 vit_r8 byd_cB es_s0
```
The printed table / compare.json include MAE / RMSE / AbsRel / delta1 / near-mid-far bands and a
`Params(M)` column.

## Note
Real training/eval require the RAD cache (`/root/implicit_full_cache/ic2_256x512_nolog`:
0deg spec + depth + mask). Waveform models also read `../cache/rx_wave`.
