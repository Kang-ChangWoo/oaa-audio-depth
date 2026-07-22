"""EchoDiffusionDepth -- port of EchoDiffusion's "econet" (Zhang et al.) depth model.

SD-backbone, DETERMINISTIC audio-to-depth baseline (single forward, no diffusion
sampling loop).  Architecture (faithful to the original repo `models/ecoNet.py`):

    spec  --(resize 128x256)-->  UNet_aspp_asff  --(1x1 adapter 128->512)-->  latents
    wave  -->  CIDE (wav2vec2 -> class-prob -> learned embeddings)          -->  cross-attn context
    latents + context  -->  Stable-Diffusion UNet (FROZEN-STRUCTURE feature
                            extractor, run once at fixed t=1, its `.out` head
                            deleted -> returns hierarchical features)
                        -->  EcoDepth encoder head  -->  Decoder (x32 upsample)
                        -->  depth head  -->  sigmoid  -->  (B,1,256,512) in [0,1]

The ×10 m metric scaling is applied later by the trainer, so this module outputs a
normalized depth in [0,1] (sigmoid only -- NOT ×max_depth as the original did).

------------------------------------------------------------------------------
REQUIRES the isolated conda env + weights (this will NOT run in the base env):

    conda activate /root/local1/changwoo/echodiff_env

    Key package versions: python 3.10, torch 1.13.1+cu117, torchvision 0.14.1,
    mmcv-full 1.7.1, pytorch-lightning 1.9.5, transformers 4.25.1,
    diffusers 0.11.1, kornia 0.6.9, timm 0.6.13, omegaconf 2.3.0, einops 0.6.1,
    numpy 1.24.4, setuptools 69.5.1, taming-transformers-rom1504, openai-clip.

    Weights (HuggingFace cache):
        facebook/wav2vec2-base-960h  ->  /root/local1/changwoo/_echodiff_weights
    Set HF_HOME to that dir (this module defaults it if unset).  No Stable
    Diffusion checkpoint is needed: the SD UNet here is a *custom* small config
    (model_channels=32, in_channels=512, z_channels=2) that no real SD-v1
    checkpoint is compatible with -- the original repo never loads one either
    (its `sd_path` arg is unused).  CLIP weights are likewise NOT needed because
    the CLIP text embedder / VAE of LatentDiffusion are unused by this model's
    forward; we instantiate only the SD UNet.
------------------------------------------------------------------------------

Interface:
    m = EchoDiffusionDepth(in_ch=2)
    depth = m(spec, wave)          # spec (B,in_ch,256,512), wave (B,2,3200)
                                   # -> depth (B,1,256,512) in [0,1]
"""

import os
import sys

# --- make the vendored SD (`ldm`) + eco (`eco`) source importable ------------
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "echodiffusion_src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Default the HF cache to where the weights were downloaded, unless the caller
# already set it.  wav2vec2-base-960h is expected under this dir.
os.environ.setdefault("HF_HOME", "/root/local1/changwoo/_echodiff_weights")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import Wav2Vec2Model
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer

from ldm.util import instantiate_from_config
from eco.models_eco import UNetWrapper, EmbeddingAdapter
from eco.ASPP_ASFF import UNet_aspp_asff

# Path to the (custom, small) SD-UNet config vendored alongside this file.
_UNET_CFG = os.path.join(_SRC, "v1-inference.yaml")

# Spatial size fed to the aspp_asff encoder.  The original resized the spec to a
# square 128x128; we use a 1:2 (128x256) resize to preserve the 256x512 target
# aspect ratio.  This makes the encoder emit an 8x16 feature map, which the
# unchanged Decoder (x32) upsamples to a NATIVE 256x512 depth map (fairness:
# no post-hoc upsampling of a tiny square).  The encoder input is deliberately
# slim -- allowed, since the spec width is fake-resolution -- while the OUTPUT
# is a genuine 256x512 decode.
_ENC_HW = (128, 256)


class _DiffusionWrapper(nn.Module):
    """Minimal stand-in for ldm...ddpm.DiffusionWrapper (crossattn path only).

    We instantiate ONLY the SD UNet from `unet_config` -- not the full
    LatentDiffusion -- because (a) the repo's cond_stage FrozenCLIPEmbedder is
    configured with an empty ``version`` string and crashes on instantiation,
    and (b) the VAE / CLIP text encoder are never used by EcoDepth's forward.
    """

    def __init__(self, unet_config, conditioning_key="crossattn"):
        super().__init__()
        self.diffusion_model = instantiate_from_config(unet_config)
        self.conditioning_key = conditioning_key
        assert conditioning_key == "crossattn"

    def forward(self, x, t, c_concat=None, c_crossattn=None):
        cc = torch.cat(c_crossattn, 1)
        return self.diffusion_model(x, t, context=cc)


class CIDE(nn.Module):
    """Waveform -> cross-attention context (wav2vec2 branch).  Faithful copy of
    ecoNet.CIDE.  wav2vec2 is used as a frozen feature extractor (no_grad)."""

    def __init__(self, emb_dim):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec.freeze_feature_extractor()
        self.conv = nn.Conv1d(2, 1, kernel_size=1)
        self.fc = nn.Sequential(nn.Linear(768, 400), nn.GELU(), nn.Linear(400, 100))
        self.dim = emb_dim
        self.m = nn.Softmax(dim=1)
        self.embeddings = nn.Parameter(torch.randn(100, self.dim))
        self.embedding_adapter = EmbeddingAdapter(emb_dim=self.dim)
        self.gamma = nn.Parameter(torch.ones(self.dim) * 1e-4)

    def forward(self, x):
        x = self.conv(x)          # (B,2,T) -> (B,1,T)
        x = x.squeeze(1)          # (B,T)

        # wav2vec2 needs a minimum sequence length for its masking / conv stack.
        mask_length = 10
        min_input_length = self.wav2vec.config.inputs_to_logits_ratio * mask_length * 2
        if x.shape[1] < min_input_length:
            x = F.pad(x, (0, min_input_length - x.shape[1]))

        with torch.no_grad():
            wav2vec_output = self.wav2vec(x).last_hidden_state  # (B,L,768)

        wav2vec_output = wav2vec_output.mean(dim=1)             # (B,768)
        class_probs = self.m(self.fc(wav2vec_output))          # (B,100)
        class_embeddings = class_probs @ self.embeddings       # (B,emb_dim)
        conditioning_scene_embedding = self.embedding_adapter(
            class_embeddings, self.gamma
        )                                                       # (B,1,emb_dim)
        return conditioning_scene_embedding


class EchoDiffusionEncoder(nn.Module):
    """Port of ecoNet.EcoDepthEncoder with the known repo bugs fixed:
      (a) 1x1 conv adapter 128->512 between aspp_asff and the SD UNet (aspp_asff
          emits 128 ch but the SD-UNet config wants in_channels=512);
      (b) no bogus ``dataset=`` kwarg;
      (c) the SD-UNet config is loaded from a resolved absolute path;
      and we instantiate only the SD UNet (see _DiffusionWrapper docstring).
    """

    def __init__(self, in_ch=2, out_dim=1024, ldm_prior=(32, 64, 256), emb_dim=768):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )
        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )
        self.apply(self._init_weights)

        self.cide_module = CIDE(emb_dim)

        # spec -> latents branch (in_conv now honors in_ch, see ASPP_ASFF.py)
        self.aspp_asff = UNet_aspp_asff(in_channels=in_ch)
        # BUG-FIX (a): adapt aspp_asff's 128-ch output to the SD-UNet in_channels=512.
        self.latent_adapter = nn.Conv2d(128, 512, kernel_size=1)

        # SD UNet feature extractor (custom small config, randomly initialized).
        config = OmegaConf.load(_UNET_CFG)
        sd_unet = _DiffusionWrapper(config.model.params.unet_config, "crossattn")
        self.unet = UNetWrapper(sd_unet, use_attn=False)
        # Delete the final projection head: we want hierarchical features, not a
        # denoised latent.  register_hier_output (inside UNetWrapper) already
        # bypasses `.out`, and deleting frees its params.
        del self.unet.unet.diffusion_model.out

    def _init_weights(self, m):
        from timm.models.layers import trunc_normal_
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, audio_spec, audio_wave):
        latents = self.aspp_asff(audio_spec)          # (B,128,h/4,w/4)
        latents = self.latent_adapter(latents)        # (B,512,h/4,w/4)

        conditioning_scene_embedding = self.cide_module(audio_wave)

        # Fixed timestep t=1: the SD UNet is a deterministic feature extractor.
        t = torch.ones((audio_spec.shape[0],), device=audio_spec.device).long()
        outs = self.unet(latents, t, c_crossattn=[conditioning_scene_embedding])

        feats = [
            outs[0],
            outs[1],
            torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1),
        ]
        x = torch.cat(
            [self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1
        )
        return self.out_layer(x)


class Decoder(nn.Module):
    """Faithful copy of ecoNet.Decoder.  Total upsampling = x32 (three deconv
    stride-2 layers = x8, then two bilinear x2 = x4).  With an 8x16 encoder
    feature map this yields a NATIVE 256x512 output."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = 3
        self.in_channels = in_channels

        self.deconv_layers = self._make_deconv_layer(3, [32, 32, 32], [2, 2, 2])

        conv_layers = [
            build_conv_layer(
                dict(type="Conv2d"), in_channels=32, out_channels=out_channels,
                kernel_size=3, stride=1, padding=1,
            ),
            build_norm_layer(dict(type="BN"), out_channels)[1],
            nn.ReLU(inplace=True),
        ]
        self.conv_layers = nn.Sequential(*conv_layers)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, conv_feats):
        out = self.deconv_layers(conv_feats[0])
        out = self.conv_layers(out)
        out = self.up(out)
        out = self.up(out)
        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type="deconv"), in_channels=in_planes, out_channels=planes,
                    kernel_size=kernel, stride=2, padding=padding,
                    output_padding=output_padding, bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes
        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            return deconv_kernel, 1, 0
        elif deconv_kernel == 3:
            return deconv_kernel, 1, 1
        elif deconv_kernel == 2:
            return deconv_kernel, 0, 0
        raise ValueError(f"Not supported num_kernels ({deconv_kernel}).")


class EchoDiffusionDepth(nn.Module):
    """Deterministic SD-backbone audio->depth model.

    Args:
        in_ch: number of magnitude-STFT channels in the spec input (2/4/6/8).

    forward(spec, wave):
        spec: (B, in_ch, 256, 512) magnitude STFT
        wave: (B, 2, 3200) binaural waveform
        returns: (B, 1, 256, 512) depth in [0, 1]
    """

    def __init__(self, in_ch=2):
        super().__init__()
        self.in_ch = in_ch

        embed_dim = 192
        channels_in = embed_dim * 8   # 1536
        channels_out = embed_dim      # 192

        self.encoder = EchoDiffusionEncoder(in_ch=in_ch, out_dim=channels_in)
        self.decoder = Decoder(channels_in, channels_out)

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, spec, wave):
        # Slim, aspect-preserving resize for the encoder (128x256).  Output stays
        # native 256x512 via the decoder -- see module header.
        if spec.shape[-2:] != _ENC_HW:
            spec_in = F.interpolate(
                spec, size=_ENC_HW, mode="bilinear", align_corners=False
            )
        else:
            spec_in = spec

        conv_feats = self.encoder(spec_in, wave)   # (B,1536,8,16)
        out = self.decoder([conv_feats])           # (B,192,256,512)
        out_depth = self.last_layer_depth(out)     # (B,1,256,512)

        # BUG-FIX (d): sigmoid only -> [0,1]; the ×metric scaling is applied by
        # the trainer, not here.
        out_depth = torch.sigmoid(out_depth)
        return out_depth
