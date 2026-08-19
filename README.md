# OAA — audio-only panoramic depth from multiple posed binaural observations

Reference implementation of **OAA** (orientation-aligned alternating attention): a model that predicts
a listener-centred equirectangular (ERP) depth map (256×512, up to 10 m) from the magnitude
spectrograms of several *posed* ear-specific echo observations (receiver yaw + ear identity known),
plus the comparison baselines and the evaluation / analysis scripts used in our experiments.

Observations are encoded **independently** by a shared encoder (not channel-stacked), conditioned
on time–frequency position and pose (AdaLN), fused by alternating intra-/inter-observation
attention, and read out by 16×32 ray-conditioned panoramic queries through a geometry-aware
cross-attention (per-(ray, observation) bias from the ray expressed in each receiver's local frame
and the ear axis), refined by self-attention and decoded to 256×512. Loss: masked L1 on
depth/10 m. 15.35 M parameters for any number of observations.

This code base is anonymised for review; names, paths and dates in comments refer to internal runs.

## Layout

```
core/                     shared library: data-module selection, metrics, checkpoint rebuild, evaluate
model/                    oaa.py (the model) + baselines (batvision, pretrained ResNet/ViT, echoscan, beyond_i2d,
                          echodiffusion + vendored echodiffusion_src/, see NOTICE.md)
data_0422.py  data_mp3d.py   Replica / Matterport3D data modules (DATA_MODULE=data_0422 | data_mp3d)
splits/mp3d/              Matterport3D scene-disjoint split keys
train_oaa.py  train_batvision.py  train_baseline.py  train_echodiffusion.py     trainers
eval.py  eval_echodiffusion.py                                                 test evaluation -> compare*.json
analysis/                 ablation / robustness evaluations (Tables 3-5, supp.)
viz/                      figures (qualitative grids, observation attribution, trajectory sequences)
tools/                    result tables, config export, optional caches
configs/{replica,mp3d}/   exact args of every final run        comparison/ comparison_mp3d/  run dirs + result JSON
```
See **SCRIPTS.md** for a per-script index (purpose, inputs/outputs, which table/figure it produces).
Scripts depend only on `core/` and `model/`, never on each other; `analysis/`, `viz/`, `tools/` scripts
chdir to the repo root on start so `comparison/...` paths resolve from anywhere.

## Environment

```
pip install -r requirements.txt                 # torch 2.8 / CUDA 12.8, Python 3.10
pip install -r requirements-echodiffusion.txt   # SEPARATE env for the EchoDiffusion baseline only (torch 1.13 + mmcv)
```

## Data

Both benchmarks are rendered with Habitat-Sim + SoundSpaces 2.0 (100 navigable positions per scene,
binaural RIR at headings 0/90/180/270°, 3 ms sweep, 48 kHz, radial ERP depth). Expected layout per scene:

```
<ROOT>/<scene>/audio_wav/audio_NNN.wav          stereo 48 kHz echo (index = 4*pos + heading_slot)
<ROOT>/<scene>/erp_depth[_radial]/erp_depth_NNN.npy   float32 metres (512x1024), 0 = invalid
<ROOT>/<scene>/locations.json                   position / quaternion / pos_idx / yaw_deg per index
```

* Replica: `REPLICA_ROOT=/path/replica DATA_MODULE=data_0422` (scene-disjoint split `R0422_SPLIT=off3`:
  12 train / val {apartment_1, frl_apartment_4, office_3} / test {apartment_2, frl_apartment_5, office_4}).
* Matterport3D: `MP3D_ROOT=/path/mp3d DATA_MODULE=data_mp3d` (72/9/9 scenes; keys in `splits/mp3d/`).

Input recipe (identical for all methods): waveform cropped to the 10 m round-trip window
(2799 samples on Replica, 2823 on MP3D), magnitude STFT (n_fft 512, win 400, hop 160, Hann),
nearest-resized to 256×512. Depth is nearest-resized to 256×512, clipped at 10 m and normalised.
Channel order of the 8 observations: `[0L, 0R, 90L, 90R, 180L, 180R, 270L, 270R]`; modes
`r2=[0L,0R]`, `cB=[0L,0R,90R,270L]` (4 obs.), `r6=[0,90,270]×[L,R]`, `r8` = all.
Optional bit-identical spectrogram caches: `tools/build_spec_cache_{replica,mp3d}.py` (+ `REPLICA_SPEC_CACHE` / `MP3D_SPEC_CACHE`).

## Train

```bash
# OAA, Replica, 8 observations (the reported recipe; lr 5e-4, 40 ep, effective batch 33, token masking)
DATA_MODULE=data_0422 python train_oaa.py --run-name oaa_r8 --nviews 8 --lr 5e-4 --warmup-ep 4 \
    --epochs 40 --batch-size 3 --accum 11 --subset-aug --vdrop-kmax 4 --out-dir comparison
# 2 / 4 / 6 observations: --nviews 2|4|6 (no --subset-aug); see configs/*.json for every reported run
DATA_MODULE=data_0422 python train_batvision.py --run-name bat_r8 --mode r8
DATA_MODULE=data_0422 python train_baseline.py --model resnet|vit|echoscan|beyond --run-name rn_r8 --mode r8
DATA_MODULE=data_0422 <echodiff_env>/bin/python train_echodiffusion.py --run-name eco_r8 --mode r8
```
Module ablations: `--no-tf-pe --no-pose-emb --no-ray-emb --no-geo-bias --no-cross`;
input-cue ablations (retraining): `--pose-blind`, `--ear-blind`. Multi-GPU: `torchrun --nproc_per_node=N train_oaa.py ...`.

`configs/` holds the exact argument set (`train_done.json["args"]`) of every run behind a reported
number, for both datasets.

## Evaluate

```bash
DATA_MODULE=data_0422 python eval.py --run-name oaa_r8 bat_r8 rn_r8 vit_r8 es_r8      # -> comparison/compare.json
DATA_MODULE=data_0422 <echodiff_env>/bin/python eval_echodiffusion.py --run-name eco_r8 # -> comparison/compare_eco.json
```
Metrics are cos-latitude-weighted and per-image: MAE, RMSE, AbsRel, log10, δ1/δ2/δ3 (+ near/mid/far
MAE bands, parameter count). `eval.py` rebuilds each model from the args stored in its checkpoint.

Analyses: `analysis/blind.py` / `analysis/eardrop.py` (remove the pose / ear cue at inference),
`analysis/controls.py` (audio-shuffle / pose-shuffle / L-R swap controls), `analysis/micdrop*.py`
(observation subsets), `analysis/rot30.py` / `analysis/rot_jitter.py` (unseen receiver headings; needs the
rotated test set), `viz/grid*.py` / `viz/full*.py` (qualitative grids), `viz/mic_attr*.py`
(observation attribution), `viz/forward_seq.py` (trajectory sequences). Full index: SCRIPTS.md.

## Checkpoints

`comparison/<run>/best.pth` = `{"state_dict": EMA weights, "args": {...}}`; `last.pth` additionally
holds the raw weights / optimiser / scheduler for `--resume`. Final-selection runs are named `*_fin`
(Replica) — see `comparison_mp3d/FIN_SELECTION.md` for the Matterport3D selection.

## Notes

* The OAA class in `model/oaa.py` is a cleaned version of the research model: research-only
  branches were removed, and the release was verified to load every final/ablation checkpoint
  with `strict=True` and to reproduce their outputs bit-for-bit.
* EchoDiffusion only: the vendored `ldm`/`eco` code (see `model/echodiffusion_src/NOTICE.md`) needs
  the separate environment and `HF_HOME` pointing at a cache containing `facebook/wav2vec2-base-960h`.
