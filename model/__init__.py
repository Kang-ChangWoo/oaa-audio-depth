"""Model zoo for the audio-only ERP-depth comparison.

  ours : OAAv2Depth        (oaa.py)          — orientation-aligned attention (SOTA)
  base : RotDepth          (batvision.py)    — BatVision channel-stacked UNet

  Comparison baselines (ported from prior work, adapted to spec (B,in_ch,256,512) ->
  depth (B,1,256,512) in [0,1]):
    PretrainedResNet / PretrainedViT  (pretrained.py)  — ImageNet backbone, 2->3ch adapter
    BeyondI2DDepth                    (beyond_i2d.py)   — Parida CVPR'21, vision branch fed a constant
    EchoScanDepth                     (echoscan.py)     — Yeon 1D-waveform encoder + depth decoder
    EchoDiffusionDepth                (echodiffusion.py)— Zhang SD-backbone econet (deterministic)

Every model exposes forward(spec) -> depth in [0,1], except waveform/dual-input models
which document their extra inputs in their own module.
"""
from .oaa import OAAv2Depth
from .batvision import RotDepth
from .pretrained import PretrainedResNet, PretrainedViT
from .beyond_i2d import BeyondI2DDepth
from .echoscan import EchoScanDepth

__all__ = [
    "OAAv2Depth", "RotDepth",
    "PretrainedResNet", "PretrainedViT", "BeyondI2DDepth", "EchoScanDepth",
]

# EchoDiffusionDepth (echodiffusion.py) is intentionally NOT imported here: it requires a
# separate isolated conda env (ldm / Stable-Diffusion / mmcv / wav2vec2) and pretrained weights.
# Import it directly from that env: `from model.echodiffusion import EchoDiffusionDepth`.

# spec-input models take forward(spec (B,in_ch,256,512)); waveform models take forward(wave):
SPEC_MODELS = {"resnet": PretrainedResNet, "vit": PretrainedViT, "beyond": BeyondI2DDepth,
               "oaa": OAAv2Depth, "batvision": RotDepth}
WAVE_MODELS = {"echoscan": EchoScanDepth}
