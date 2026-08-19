# Third-party code vendored for the EchoDiffusion baseline

This directory is NOT our code. It is imported only by `model/echodiffusion.py` (the
EchoDiffusion comparison baseline) and is kept here so that baseline runs in an isolated
torch-1.13 environment without further installs. Nothing else in this repository depends on it.

| Path | Origin | License |
|---|---|---|
| `ldm/` | Stable Diffusion `ldm` package, CompVis (https://github.com/CompVis/stable-diffusion), itself derived from latent-diffusion / openai guided-diffusion | CreativeML Open RAIL-M (code portions MIT); see the upstream repository |
| `eco/` | EcoDepth (https://github.com/Aradhye2002/EcoDepth) `models_eco.py` / `ASPP_ASFF.py` / `util.py`, as re-used by EchoDiffusion (Zhang et al., AAAI 2025, `models/ecoNet.py`) | see the upstream repositories |
| `v1-inference.yaml` | reduced SD-UNet config (model_channels=32, in_channels=512) used by EchoDiffusion's deterministic network | as above |

Only `ldm.util`, `ldm.modules.diffusionmodules.{openaimodel,util}` and `ldm.modules.attention` are
used at runtime; the remaining `ldm/` files are carried along unchanged (some import `taming`, which
is not vendored, and will raise ImportError if imported directly).

All rights remain with the original authors. If you redistribute this directory, keep this notice
and comply with the upstream licenses.
