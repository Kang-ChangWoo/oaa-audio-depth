"""0820 experiment: pluggable per-observation audio encoders for OAA (Audio Foundation Model backbones).

Replaces ONLY the coarse per-observation encoder of OAA with a pretrained Audio Foundation Model
(AFM); everything else (fine CNN skip path, TF PE, pose/ear embedding, alternating attention,
ray-mic geometric cross-attention, decoder, loss) is the unmodified OAAv2Depth code.

Contract (drop-in for model.oaa.ViewEncoder):
    forward(x: [B*N, 1, 256, 512] magnitude spectrogram)
        -> tokens [B*N, lh*lw=512, C], fine [B*N, 4*lh*lw, fine_ch]

All backbones are ViT-B/16 encoders (dim 768, depth 12, heads 12). The original AFM patch embed
(trained on long log-mel AudioSet clips) is REPLACED by a task-specific Conv2d(1,768,16,16) patch
embedding trained from scratch on the 256(F)x512(T) echo spectrogram -> 16x32 = 512 patch tokens
(freq-major rows, same flatten order as the CNN encoder tokens). Pretrained transformer blocks,
CLS token, pre/final norms and positional embeddings are loaded from the official checkpoints;
positional embeddings are 2-D bicubic-interpolated from the model's native patch grid to 16x32
with the time/frequency axes preserved (see _SPECS[...]["layout"]).

Native grids (rows x cols of the pretrained pos-embed):
    audiomosaic  (64 time, 8 freq)   spec 1024x128, layout "tf"  (learnable pos, CLS row 0)
    bat          (64 time, 8 freq)   spec 1024x128, layout "tf"  (fixed sincos, no CLS row)
    eat / sslam  (64 time, 8 freq)   img 1024x128 (positions stored for 768x8; sliced), "tf"
    m2d          (5 freq, 62 time)   spec 80x1001, layout "ft"   (M2D-CLAP 2025, CLS row 0)
    m2d_plain    (5 freq, 38 time)   spec 80x608,  layout "ft"

The AFM input is log1p(magnitude) standardized per sample (mean/std over the whole spectrogram);
the fine CNN path receives the raw magnitude spectrogram exactly as before. No pose/yaw/ear
information reaches the AFM.

Checkpoints (never stored in git; downloaded to $AFM_WEIGHTS, default /root/local1/changwoo/_afm_weights):
    audiomosaic  hf:hanxunh/AudioMosaic-vit-b16-pretrained        (self-supervised, AudioSet-2M)
    bat          hf:lrauch/BAT-vit-b16-pretrainedAS2M             (gated-attention post-norm ViT)
    eat          hf:worstchan/EAT-base_epoch30_pretrain           (data2vec2 post-norm AltBlock)
    sslam        hf:ta012/SSLAM_pretrain                          (EAT-compatible)
    m2d          m2d/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth   (backbone.*)
    m2d_plain    m2d/m2d_vit_base-80x608p16x16-221006-mr7_enconly/checkpoint-300.pth
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.oaa import OAAv2Depth, ViewEncoder, LH, LW

_SR = 48000          # both datasets render at 48 kHz (data_0422.py / data_mp3d.py SR)
AFM_DIR = os.environ.get("AFM_WEIGHTS", "/root/local1/changwoo/_afm_weights")
BACKBONES = ("cnn", "audiomosaic", "bat", "eat", "sslam", "m2d", "m2d_plain", "m2d20ms")

# style: block forward order; layout: pos-embed native grid axis order ("tf" rows=time / "ft" rows=freq)
_SPECS = {
    "audiomosaic": dict(hf="hanxunh/AudioMosaic-vit-b16-pretrained", prefix="", style="prenorm",
                        grid=(64, 8), layout="tf", pos_key="pos_embed", pos_has_cls=True,
                        cls_key="cls_token", pre_norm_key="norm_pre", final_norm_key=None, cls_gets_pos=True),
    "bat": dict(hf="lrauch/BAT-vit-b16-pretrainedAS2M", prefix="", style="postnorm_gate",
                grid=(64, 8), layout="tf", pos_key="pos_embed", pos_has_cls=False,
                cls_key="cls_token", pre_norm_key="pre_norm", final_norm_key=None, cls_gets_pos=False,
                patch_key="patch_embed.proj"),
    "eat": dict(hf="worstchan/EAT-base_epoch30_pretrain", prefix="model.", style="postnorm_alt",
                grid=(64, 8), layout="tf", pos_key="fixed_positional_encoder.positions", pos_has_cls=False,
                cls_key="extra_tokens", pre_norm_key="pre_norm", final_norm_key=None, cls_gets_pos=False,
                patch_key="local_encoder.proj"),
    "sslam": dict(hf="ta012/SSLAM_pretrain", prefix="model.", style="postnorm_alt",
                  grid=(64, 8), layout="tf", pos_key="fixed_positional_encoder.positions", pos_has_cls=False,
                  cls_key="extra_tokens", pre_norm_key="pre_norm", final_norm_key=None, cls_gets_pos=False,
                patch_key="local_encoder.proj"),
    "m2d": dict(pth="m2d/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth",
                prefix="backbone.", style="prenorm", grid=(5, 62), layout="ft", pos_key="pos_embed",
                pos_has_cls=True, cls_key="cls_token", pre_norm_key=None, final_norm_key="norm", cls_gets_pos=True),
    "m2d_plain": dict(pth="m2d/m2d_vit_base-80x608p16x16-221006-mr7_enconly/checkpoint-300.pth",
                      prefix="", style="prenorm", grid=(5, 38), layout="ft", pos_key="pos_embed",
                      pos_has_cls=True, cls_key="cls_token", pre_norm_key=None, final_norm_key="norm",
                      cls_gets_pos=True),
    # 20 ms temporal-resolution M2D-CLAP: native patch 80(freq)x2(time) -> tokens are ~pure time slices.
    # Our task patch (128, 2): grid (2 freq, 256 time) = 512 tokens, 8x finer time resolution than 16x16.
    "m2d20ms": dict(pth="m2d/m2d_clap_vit_base-80x1001p80x2p16kpBpTI-2025/checkpoint-30.pth",
                    prefix="backbone.", style="prenorm", grid=(1, 500), layout="ft", pos_key="pos_embed",
                    pos_has_cls=True, cls_key="cls_token", pre_norm_key=None, final_norm_key="norm",
                    cls_gets_pos=True, patch=(128, 2)),
}


# --------------------------------------------------------------------------- ViT parts
class _Attn(nn.Module):
    """Standard ViT attention; optional BAT output gate (attn_out * sigmoid(gate(x_in)))."""
    def __init__(self, dim=768, heads=12, gate=False):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim) if gate else None

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, D)
        if self.gate is not None:
            o = o * torch.sigmoid(self.gate(x))
        return self.proj(o)


class _Block(nn.Module):
    """One ViT-B block in the exact forward order of the source model.
       prenorm       (timm/MAE: AudioMosaic, M2D): x += attn(n1(x)); x += mlp(n2(x))
       postnorm_gate (BAT):                        x = n1(x + gated_attn(x)); x = n2(x + mlp(x))
       postnorm_alt  (EAT/SSLAM data2vec2):        x += attn(x); r = n1(x); x = n2(r + mlp(r))"""
    def __init__(self, style, dim=768, heads=12, mlp_ratio=4.0):
        super().__init__()
        self.style = style
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attn(dim, heads, gate=(style == "postnorm_gate"))
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        if self.style == "prenorm":
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))
        if self.style == "postnorm_gate":
            x = self.norm1(x + self.attn(x))
            return self.norm2(x + self.mlp(x))
        x = x + self.attn(x)                       # postnorm_alt
        r = self.norm1(x)
        return self.norm2(r + self.mlp(r))


# --------------------------------------------------------------------------- checkpoint loading
def _load_source_sd(name):
    """-> (flat state dict of the AFM encoder, checkpoint identifier string)."""
    s = _SPECS[name]
    if "hf" in s:
        from huggingface_hub import hf_hub_download
        os.environ.setdefault("HF_HOME", AFM_DIR)
        p = hf_hub_download(s["hf"], "model.safetensors")
        from safetensors.torch import load_file
        sd, ident = load_file(p), f"hf:{s['hf']}"
    else:
        p = os.path.join(AFM_DIR, s["pth"])
        if not os.path.exists(p):
            raise FileNotFoundError(f"AFM checkpoint missing: {p} (download the official release first)")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd, ident = ck.get("model", ck), s["pth"]
    pre = s["prefix"]
    return {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, ident


def _interp_pos(pos, src_grid, layout, gh, gw):
    """2-D bicubic interpolation of a patch pos-embed (src_grid rows x cols, D) to the target
    (gh freq x gw time) token grid, preserving axis semantics, returned in OUR freq-major layout (gh*gw, D)."""
    D = pos.shape[-1]
    g = pos.reshape(1, *src_grid, D).permute(0, 3, 1, 2).float()          # (1, D, rows, cols) native order
    tgt = (gw, gh) if layout == "tf" else (gh, gw)                        # native-order target (rows, cols)
    g = F.interpolate(g, size=tgt, mode="bicubic", align_corners=False)
    if layout == "tf":                                                    # (1,D,T,F) -> (1,D,F,T)
        g = g.permute(0, 1, 3, 2)
    return g.flatten(2).transpose(1, 2).reshape(1, gh * gw, D)            # freq-major rows, like our patch conv


# --------------------------------------------------------------------------- conv-stem zoo (A)
# Every stem maps (B,1,256,512) -> (B,768,16,32): the same token grid as the 16x16 linear patch, so
# the pretrained pos-embed, the CLS handling and everything downstream are untouched. Only the way the
# 16x downsample is spent differs. Parameter counts (vs A0 "conv" = 2.139M) are asserted in the tests.


class _ResTempStem(nn.Module):
    """A1 residual temporal stem: one stride-2 3x5 entry, a stride-1 residual 3x5 pair (longer receptive
    field along TIME than frequency -- echo structure is a time series per frequency band), then three
    stride-2 3x3 convs to the ViT width. +5.7% params over A0."""
    def __init__(self, dim, ch0=64):
        super().__init__()
        self.c1 = nn.Conv2d(1, ch0, (3, 5), 2, (1, 2))
        self.n1 = nn.GroupNorm(8, ch0)
        self.r1 = nn.Conv2d(ch0, ch0, (3, 5), 1, (1, 2))
        self.nr = nn.GroupNorm(8, ch0)
        self.r2 = nn.Conv2d(ch0, ch0, (3, 5), 1, (1, 2))
        self.rest = nn.Sequential(nn.Conv2d(ch0, 128, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(128, 256, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(256, dim, 3, 2, 1))
        nn.init.zeros_(self.r2.weight); nn.init.zeros_(self.r2.bias)       # residual branch starts as identity

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = h + self.r2(F.gelu(self.nr(self.r1(h))))
        return self.rest(h)


class _MultiScaleStem(nn.Module):
    """A2 two-branch multi-scale stem: after one stride-2 entry, a plain 3x3 branch (early/local echo)
    and a temporally dilated 3x3 branch (slightly longer temporal structure) are concatenated and
    projected back. Exactly two branches, no Inception fan-out. +3.8% params over A0."""
    def __init__(self, dim, ch0=64):
        super().__init__()
        self.c1 = nn.Conv2d(1, ch0, 3, 2, 1)
        self.b_local = nn.Conv2d(ch0, ch0, 3, 1, 1)
        self.b_dilat = nn.Conv2d(ch0, ch0, 3, 1, padding=(1, 2), dilation=(1, 2))   # dilate TIME only
        self.merge = nn.Conv2d(2 * ch0, ch0, 1)
        self.rest = nn.Sequential(nn.Conv2d(ch0, 128, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(128, 256, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(256, dim, 3, 2, 1))

    def forward(self, x):
        h = F.gelu(self.c1(x))
        h = self.merge(torch.cat([self.b_local(h), self.b_dilat(h)], 1))
        return self.rest(F.gelu(h))


class _FactorizedStem(nn.Module):
    """A3 factorized time/frequency stem: a 3x1 frequency conv then a 1x5 temporal conv replace the first
    square conv, separating the two roles at a lower parameter cost. +1.0% params over A0."""
    def __init__(self, dim, ch0=64):
        super().__init__()
        self.f = nn.Conv2d(1, ch0, (3, 1), (2, 1), (1, 0))
        self.t = nn.Conv2d(ch0, ch0, (1, 5), (1, 2), (0, 2))
        self.rest = nn.Sequential(nn.Conv2d(ch0, 128, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(128, 256, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(256, dim, 3, 2, 1))

    def forward(self, x):
        return self.rest(F.gelu(self.t(F.gelu(self.f(x)))))


def _plain_conv_stem(dim, patch=(16, 16)):
    """A0 (released): 4 x stride-2 3x3 convs, trailing GELU dropped (linear-out like the ViT stem).

    `patch` is the total (freq, time) downsampling the stem must produce. The released 16x16 is four
    stride-(2,2) steps. A non-square patch (the token-allocation control, e.g. 8x32) is reached by
    redistributing the SAME four steps per axis -- 8 = 2,2,2,1 and 32 = 4,2,2,2 -- so depth, channel
    schedule and kernel size are unchanged and the stem's parameter count stays identical. Only where
    the strides fall differs, which is exactly the variable under test.
    """
    import math as _m

    def sched(p):
        k = int(_m.log2(p))
        assert 2 ** k == p, f"patch side {p} must be a power of two"
        assert 0 <= k <= 8, f"patch side {p} out of range"
        st = [1, 1, 1, 1]                                    # spread k halvings over exactly 4 layers
        for i in range(k):
            st[3 - (i % 4)] *= 2
        return st

    sh, sw = sched(patch[0]), sched(patch[1])
    ch = [1, 64, 128, 256, dim]
    layers = []
    for i in range(4):
        layers += [nn.Conv2d(ch[i], ch[i + 1], 3, (sh[i], sw[i]), 1), nn.GELU()]
    return nn.Sequential(*layers[:-1])


def _mel_fb(n_mels, n_freq, sr, fmax=None):
    """HTK triangular mel filterbank (n_mels, n_freq) over the linear STFT bins we already cache.

    The AFM checkpoints were pretrained on log-MEL patches, so feeding them linear-frequency bins
    puts the pretrained patch embedding out of distribution before a single block runs. This is the
    filterbank that puts the input back in the format the pretrained weights expect."""
    fmax = fmax or sr / 2
    m = lambda f: 2595.0 * math.log10(1.0 + f / 700.0)
    mi = lambda x: 700.0 * (10.0 ** (x / 2595.0) - 1.0)
    pts = torch.tensor([mi(v) for v in torch.linspace(m(0.0), m(fmax), n_mels + 2).tolist()])
    bins = pts / (sr / 2) * (n_freq - 1)
    fb = torch.zeros(n_mels, n_freq)
    idx = torch.arange(n_freq, dtype=torch.float32)
    for i in range(n_mels):
        lo, ctr, hi = bins[i], bins[i + 1], bins[i + 2]
        up = (idx - lo) / max(float(ctr - lo), 1e-6)
        dn = (hi - idx) / max(float(hi - ctr), 1e-6)
        fb[i] = torch.clamp(torch.minimum(up, dn), min=0.0)
    return fb / fb.sum(1, keepdim=True).clamp(min=1e-6)


_STEMS = {"linear": None, "native": None, "conv": _plain_conv_stem, "conv_res": _ResTempStem,
          "conv_ms": _MultiScaleStem, "conv_fact": _FactorizedStem}


class AFMBackbone(nn.Module):
    """Pretrained ViT-B/16 audio encoder with a task-specific patch embedding.
    forward(x: [B*, 1, 256, 512] magnitude spec) -> [B*, lh*lw, out_dim] patch tokens (no CLS/no pooling)."""
    DIM, DEPTH = 768, 12

    def __init__(self, name, out_dim=256, lh=LH, lw=LW, pretrained=True, verbose=True, stem="linear",
                 input_norm="std", patch_hw=None):
        super().__init__()
        assert name in _SPECS, f"unknown audio backbone {name} (choose from {list(_SPECS)})"
        assert stem in _STEMS, f"bad afm stem {stem} (choose from {list(_STEMS)})"
        # AFM input statistics: "std" = log1p + per-sample standardize (default);
        # "db" = 20*log10 (AudioSet log-mel-like) + per-sample standardize;
        # "db_minmax" = dB + per-sample min-max to [0,1] (BAT's native per_sample_minmax_after_db).
        assert input_norm in ("std", "db", "db_minmax"), f"bad afm input norm {input_norm}"
        self.input_norm = input_norm
        s = _SPECS[name]
        self.name, self.lh, self.lw, self.stem_kind = name, lh, lw, stem
        self.cls_gets_pos = s["cls_gets_pos"]
        if stem == "native":
            # Feed the backbone its OWN input format: a log-mel image at the pretrained patch grid,
            # so the pretrained patch embedding AND positional embedding transfer verbatim (no
            # interpolation, no from-scratch input layer). Requires the native grid to hold exactly
            # lh*lw tokens, which is true for the 1024x128 AudioSet backbones (64x8 = 512).
            assert s.get("patch_key"), f"{name} has no pretrained patch embed to reuse"
            assert s["grid"][0] * s["grid"][1] == lh * lw, \
                f"{name} native grid {s['grid']} != {lh * lw} tokens"
            self.grid_hw = s["grid"]                                      # (rows, cols) in NATIVE layout
            self.native_img = (16 * s["grid"][0], 16 * s["grid"][1])      # e.g. (1024 time, 128 freq)
            self.patch = nn.Conv2d(1, self.DIM, 16, 16)                   # pretrained (see _load_pretrained)
            n_mels = s["grid"][1] * 16 if s["layout"] == "tf" else s["grid"][0] * 16
            self.register_buffer("mel_fb", _mel_fb(n_mels, 256, _SR), persistent=False)
        else:
            # patch_hw overrides the (freq, time) patch size. The token COUNT must stay lh*lw, so the
            # only free choice is how those tokens are split between the two axes: (16,16) -> 16 freq x
            # 32 time (released), (8,32) -> 32 freq x 16 time, (32,8) -> 8 freq x 64 time. This is the
            # control for "is the released grid spending time tokens it has no distinct frames for?"
            # -- at hop 160 the spectrogram holds 18 real frames behind 32 time tokens.
            ph, pw = patch_hw or s.get("patch", (256 // lh, 512 // lw))    # default 16x16 on the 256x512 spec
            self.grid_hw = (256 // ph, 512 // pw)                         # token grid (freq rows, time cols)
            M = self.grid_hw[0] * self.grid_hw[1]
            assert M == lh * lw, f"patch {ph}x{pw} gives {M} tokens, OAA needs {lh * lw}"
            if stem != "linear":
                if stem == "conv":
                    self.patch = _plain_conv_stem(self.DIM, (ph, pw))     # any power-of-two patch
                else:
                    assert (ph, pw) == (16, 16), f"stem {stem} only supports the 16x16 grid"
                    self.patch = _STEMS[stem](self.DIM)
            else:
                self.patch = nn.Conv2d(1, self.DIM, (ph, pw), (ph, pw))   # NEW (task-specific, base LR)
        self.pos_embed = nn.Parameter(torch.zeros(1, lh * lw, self.DIM))  # pretrained (interpolated)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.DIM))        # pretrained
        self.pre_norm = nn.LayerNorm(self.DIM, eps=1e-6) if s["pre_norm_key"] else None
        self.blocks = nn.ModuleList([_Block(s["style"]) for _ in range(self.DEPTH)])
        self.out_norm = nn.LayerNorm(self.DIM, eps=1e-6)                  # pretrained for m2d*, NEW otherwise
        self.proj = nn.Linear(self.DIM, out_dim)                          # NEW (base LR)
        self.register_buffer("cls_pos_buf", torch.zeros(1, 1, self.DIM), persistent=True)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.checkpoint_id = "RANDOM-INIT"
        self.pretrained_loaded = False
        if pretrained:
            self._load_pretrained(s, verbose)

    # ---- pretrained transfer -------------------------------------------------
    def _load_pretrained(self, s, verbose):
        sd, ident = _load_source_sd(self.name)
        used = []

        def take(key):
            used.append(key)
            return sd[key]

        # transformer blocks (timm naming in every source checkpoint)
        for i, blk in enumerate(self.blocks):
            m = {f"norm1": blk.norm1, f"norm2": blk.norm2}
            for nm, mod in m.items():
                mod.weight.data.copy_(take(f"blocks.{i}.{nm}.weight")); mod.bias.data.copy_(take(f"blocks.{i}.{nm}.bias"))
            blk.attn.qkv.weight.data.copy_(take(f"blocks.{i}.attn.qkv.weight"))
            blk.attn.qkv.bias.data.copy_(take(f"blocks.{i}.attn.qkv.bias"))
            blk.attn.proj.weight.data.copy_(take(f"blocks.{i}.attn.proj.weight"))
            blk.attn.proj.bias.data.copy_(take(f"blocks.{i}.attn.proj.bias"))
            if blk.attn.gate is not None:
                blk.attn.gate.weight.data.copy_(take(f"blocks.{i}.attn.gate.weight"))
                blk.attn.gate.bias.data.copy_(take(f"blocks.{i}.attn.gate.bias"))
            blk.mlp[0].weight.data.copy_(take(f"blocks.{i}.mlp.fc1.weight")); blk.mlp[0].bias.data.copy_(take(f"blocks.{i}.mlp.fc1.bias"))
            blk.mlp[2].weight.data.copy_(take(f"blocks.{i}.mlp.fc2.weight")); blk.mlp[2].bias.data.copy_(take(f"blocks.{i}.mlp.fc2.bias"))
        # cls / pre-norm / final norm
        self.cls_token.data.copy_(take(s["cls_key"]).reshape(1, 1, self.DIM))
        if self.pre_norm is not None:
            self.pre_norm.weight.data.copy_(take(f"{s['pre_norm_key']}.weight"))
            self.pre_norm.bias.data.copy_(take(f"{s['pre_norm_key']}.bias"))
        if s["final_norm_key"]:
            self.out_norm.weight.data.copy_(take(f"{s['final_norm_key']}.weight"))
            self.out_norm.bias.data.copy_(take(f"{s['final_norm_key']}.bias"))
        # positional embedding: slice to the native training grid, 2-D interpolate to (lh, lw)
        pos = take(s["pos_key"]).float().reshape(-1, self.DIM)
        cls_pos = None
        if s["pos_has_cls"]:
            cls_pos, pos = pos[:1], pos[1:]
        n_native = s["grid"][0] * s["grid"][1]
        pos = pos[:n_native]                                              # EAT/SSLAM store 768x8; use first 64x8
        assert pos.shape[0] == n_native, f"pos len {pos.shape[0]} != grid {s['grid']}"
        if self.stem_kind == "native":
            self.pos_embed.data.copy_(pos.reshape(1, -1, self.DIM))       # native grid: verbatim, no interp
            self.patch.weight.data.copy_(take(f"{s['patch_key']}.weight").reshape(self.DIM, 1, 16, 16))
            self.patch.bias.data.copy_(take(f"{s['patch_key']}.bias"))
        else:
            self.pos_embed.data.copy_(_interp_pos(pos, s["grid"], s["layout"], *self.grid_hw))
        if cls_pos is not None and s["cls_gets_pos"]:
            self.cls_pos_buf.copy_(cls_pos.reshape(1, 1, self.DIM))
        self.checkpoint_id = ident
        self.pretrained_loaded = True
        if verbose:
            unused = [k for k in sd if k not in used]
            print(f"[afm] backbone={self.name} ckpt={ident}\n"
                  f"[afm]   pretrained loaded: YES ({len(used)} tensors; {len(unused)} source tensors unused, "
                  f"e.g. {unused[:4]})\n"
                  f"[afm]   native grid {s['grid']} ({s['layout']}) -> target token grid (freq {self.grid_hw[0]} x time {self.grid_hw[1]}); "
                  f"style={s['style']}; new params: "
                  f"{'proj' if self.stem_kind == 'native' else 'patch-embed, proj'}"
                  f"{'' if s['final_norm_key'] else ', out-norm'}", flush=True)

    # ---- forward -------------------------------------------------------------
    def forward(self, x):
        # AFM input normalization (the fine CNN path keeps the raw magnitude)
        if self.input_norm == "std":
            x = torch.log1p(x)
        else:                                                              # dB scale, like AudioSet log-mels
            x = 20.0 * torch.log10(x.clamp(min=1e-5))
        if self.input_norm == "db_minmax":                                 # BAT: per-sample minmax after dB
            lo = x.amin(dim=(1, 2, 3), keepdim=True)
            hi = x.amax(dim=(1, 2, 3), keepdim=True)
            x = (x - lo) / (hi - lo).clamp(min=1e-4)
        else:
            mu = x.mean(dim=(1, 2, 3), keepdim=True)
            sd = x.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-4)        # zeroed (vdrop) inputs stay finite
            x = (x - mu) / sd
        if self.stem_kind == "native":
            # (B,1,256 freq,512 time) -> log-mel -> the backbone's own (time, freq) image
            x = torch.einsum("mf,bcft->bcmt", self.mel_fb.to(x.dtype), x)  # freq 256 -> n_mels
            x = x.transpose(-2, -1)                                        # -> (B,1,time,mel)
            if _SPECS[self.name]["layout"] != "tf":
                x = x.transpose(-2, -1)
            x = F.interpolate(x, size=self.native_img, mode="bilinear", align_corners=False)
        t = self.patch(x).flatten(2).transpose(1, 2)                       # (B*, lh*lw, 768) freq-major
        t = t + self.pos_embed
        cls = self.cls_token + self.cls_pos_buf
        t = torch.cat([cls.expand(t.shape[0], -1, -1), t], 1)
        if self.pre_norm is not None:
            t = self.pre_norm(t)
        for blk in self.blocks:
            t = blk(t)
        return self.proj(self.out_norm(t[:, 1:]))                          # drop CLS -> patch tokens only

    def pretrained_param_names(self, prefix=""):
        """Names of params initialized from the AFM checkpoint (LR group 0.1x). The task-specific
        patch embed and output projection (and out_norm when not in the source ckpt) stay at base LR."""
        s = _SPECS[self.name]
        names = [f"{prefix}pos_embed", f"{prefix}cls_token"]
        names += [n for n, _ in self.named_parameters(prefix=prefix[:-1] if prefix else "")
                  if ".blocks." in n or n.startswith("blocks.")]
        if self.pre_norm is not None:
            names += [f"{prefix}pre_norm.weight", f"{prefix}pre_norm.bias"]
        if s["final_norm_key"]:
            names += [f"{prefix}out_norm.weight", f"{prefix}out_norm.bias"]
        if self.stem_kind == "native":                 # the patch embed is pretrained too in this mode
            names += [f"{prefix}patch.weight", f"{prefix}patch.bias"]
        return set(names)


# --------------------------------------------------------------------------- OAA integration
class AFMViewEncoder(nn.Module):
    """Drop-in replacement for model.oaa.ViewEncoder: AFM coarse tokens + the ORIGINAL lightweight
    CNN fine path (ViewEncoder truncated at its (2lh, 2lw) fine tap, weights fresh, base LR)."""
    def __init__(self, name, C=256, ngf=64, in_ch=1, norm="group", lh=LH, lw=LW,
                 enc_res=(256, 512), stem_stride1=False, pretrained=True, afm_stem="linear",
                 afm_input_norm="std", afm_patch=None):
        super().__init__()
        assert in_ch == 1, "AFM encoder supports 1 channel per observation"
        fe = ViewEncoder(C, ngf, in_ch, norm, lh, lw, enc_res, stem_stride1)
        stages = int(round(math.log2(enc_res[0] / lh)))
        fe.net = fe.net[: 2 * (stages - 1)]                               # keep up to the (2lh,2lw) fine tap
        self.fine_enc = fe
        self.fine_ch = fe.fine_ch
        self.afm = AFMBackbone(name, out_dim=C, lh=lh, lw=lw, pretrained=pretrained, stem=afm_stem,
                               input_norm=afm_input_norm, patch_hw=afm_patch)
        self.C, self.lh, self.lw = C, lh, lw
        self.enc_res, self.stem_stride1 = enc_res, stem_stride1

    def forward(self, x):                                                  # x: (B*, 1, 256, 512)
        fe = self.fine_enc
        want = tuple(fe.enc_res) if fe.stem_stride1 else tuple(2 * r for r in fe.enc_res)
        h = x if x.shape[-2:] == want else F.interpolate(x, size=want, mode="bilinear", align_corners=False)
        h = fe.stem(h)
        for blk in fe.net:
            h = blk(h)
        assert h.shape[-2:] == (2 * self.lh, 2 * self.lw), f"fine tap {h.shape}"
        fine = h.flatten(2).transpose(1, 2)                                # (B*, 4M, fine_ch)
        tok = self.afm(x)                                                  # (B*, M, C)
        return tok, fine


class OAAv2DepthAFM(OAAv2Depth):
    """OAAv2Depth with the coarse per-observation encoder swapped for a pretrained AFM.
    Everything downstream of the encoder is inherited unchanged, except for two optional and
    zero-initialised additions (both default OFF, so the released behaviour is bit-identical):

    mic_diff -- mic-differential SSLAM (experiment B). Token indexing is identical across mics (same
        patch grid, and no pose/yaw/ear ever reaches the AFM), so S_i - mean_j S_j is the mic-to-mic
        difference at the SAME time-frequency patch. We inject only that difference back as a residual:
          "res"      Z_i = S_i + alpha * P(D_i)                      alpha learnable, P zero-init
          "gate"     Z_i = S_i + g_i (*) P(D_i)                      channel-wise gate from pooled
                                                                     [S_i ; D_i ; pose_emb_i]
          "gate_ctx" as "gate", and additionally the common component mean_j S_j is handed to the
                     decoder ONCE as a global FiLM conditioning instead of being repeated per mic.
        The gate bias starts at -2 (sigmoid ~ 0.12) and P is zero-init, so training begins from the
        unmodified pretrained representation. Per-mic gate means are stashed in `last_gate` for eval.

    fine_res -- lightweight fine-CNN residual into the coarse stream (experiment C). The existing fine
        path currently reaches the network only once, at the decoder, and `_fine_lift` collapses the mic
        axis with a plain mean -- near-field per-mic precision is diluted there. This adds NO new
        encoder: it pools the already-computed fine tokens of each observation to the coarse token grid
        and adds a zero-init projection of them to that observation's coarse tokens.
    """
    def __init__(self, audio_backbone, afm_pretrained=True, afm_stem="linear", afm_input_norm="std",
                 mic_diff="none", fine_res=False, afm_patch=None, **kw):
        super().__init__(**kw)
        assert audio_backbone in _SPECS, f"bad audio_backbone {audio_backbone}"
        assert mic_diff in ("none", "res", "gate", "gate_ctx"), f"bad mic_diff {mic_diff}"
        self.audio_backbone = audio_backbone
        self.mic_diff, self.fine_res_on = mic_diff, bool(fine_res)
        self.last_gate = None; self.last_gate_fine = None
        self.enc = AFMViewEncoder(audio_backbone, C=self.C, in_ch=self.in_ch, lh=self.lh, lw=self.lw,
                                  enc_res=self.enc_res, stem_stride1=kw.get("stem_stride1", False),
                                  pretrained=afm_pretrained, afm_stem=afm_stem, afm_input_norm=afm_input_norm,
                                  afm_patch=afm_patch)
        if mic_diff != "none":
            self.diff_proj = nn.Linear(self.C, self.C)
            nn.init.zeros_(self.diff_proj.weight); nn.init.zeros_(self.diff_proj.bias)
            if mic_diff == "res":
                self.diff_alpha = nn.Parameter(torch.tensor(0.1))
            else:
                self.gate_mlp = nn.Sequential(nn.Linear(3 * self.C, self.C), nn.GELU(),
                                              nn.Linear(self.C, self.C))
                nn.init.zeros_(self.gate_mlp[-1].weight)
                nn.init.constant_(self.gate_mlp[-1].bias, -2.0)            # sigmoid(-2) ~ 0.12
            if mic_diff == "gate_ctx":
                self.ctx_film = nn.Linear(self.C, 2 * self.C)
                nn.init.zeros_(self.ctx_film.weight); nn.init.zeros_(self.ctx_film.bias)
        if self.fine_res_on:
            self.fine_res = nn.Linear(self.enc.fine_ch, self.C)
            nn.init.zeros_(self.fine_res.weight); nn.init.zeros_(self.fine_res.bias)
            self.fine_gate = nn.Sequential(nn.Linear(3 * self.C, self.C), nn.GELU(),
                                           nn.Linear(self.C, self.C))      # channel-wise g_L
            nn.init.zeros_(self.fine_gate[-1].weight)
            nn.init.constant_(self.fine_gate[-1].bias, -2.0)               # sigmoid(-2) ~ 0.12
        # fine_in built by super() from the full ViewEncoder's fine_ch; the truncated fine path keeps
        # the same channel count by construction — assert instead of trusting it silently.
        assert self.fine_in.in_features == self.enc.fine_ch, \
            f"fine_ch mismatch {self.fine_in.in_features} vs {self.enc.fine_ch}"

    # ---------------------------------------------------------------- experiments B / C
    def _encode(self, spec, view_pose=None):
        """OAAv2Depth._encode with the mic-differential (B) and fine-residual (C) injections inserted
        between the encoder and the positional/pose additions. With both off this is the parent verbatim."""
        assert view_pose is None or len(view_pose) == self.nv, f"view_pose len must be {self.nv}"
        assert spec.size(1) == self.nv * self.in_ch, f"expected {self.nv * self.in_ch}ch, got {spec.size(1)}"
        B = spec.size(0); dev = spec.device; H, W = spec.shape[-2:]
        pose_feat, poses = self._pose_tensors(dev, view_pose)
        v = spec.view(B, self.nv, self.in_ch, H, W).reshape(B * self.nv, self.in_ch, H, W)
        enc_t, fine_t = self.enc(v)
        t = enc_t.reshape(B, self.nv, self.M, self.C)
        fine = fine_t.reshape(B, self.nv, 4 * self.M, self.enc.fine_ch)
        self.ctx = None

        if self.mic_diff != "none":                                        # --- experiment B
            Sbar = t.mean(1, keepdim=True)                                 # (B,1,M,C) common room/reverb
            D = t - Sbar                                                   # (B,nv,M,C) mic-specific
            dproj = self.diff_proj(D)
            if self.mic_diff == "res":
                t = t + self.diff_alpha * dproj
            else:
                e = self.pose_emb(pose_feat).view(1, self.nv, self.C).expand(B, -1, -1)
                g = torch.sigmoid(self.gate_mlp(torch.cat([t.mean(2), D.mean(2), e], -1)))   # (B,nv,C)
                t = t + g.unsqueeze(2) * dproj
                self.last_gate = g.detach().mean(-1)                       # (B,nv) for eval-time analysis
            if self.mic_diff == "gate_ctx":
                self.ctx = Sbar.squeeze(1).mean(1)                         # (B,C) single global context

        if self.fine_res_on:                                               # --- experiment C
            f = fine.view(B, self.nv, 2 * self.lh, 2 * self.lw, self.enc.fine_ch)
            f = f.view(B, self.nv, self.lh, 2, self.lw, 2, self.enc.fine_ch).mean((3, 5))    # 2x2 pool
            Lp = self.fine_res(f.reshape(B, self.nv, self.M, self.enc.fine_ch))              # (B,nv,M,C)
            e = self.pose_emb(pose_feat).view(1, self.nv, self.C).expand(B, -1, -1)
            gl = torch.sigmoid(self.fine_gate(torch.cat([t.mean(2), Lp.mean(2), e], -1)))    # (B,nv,C)
            t = t + gl.unsqueeze(2) * Lp
            self.last_gate_fine = gl.detach().mean(-1)                     # (B,nv) for eval-time analysis

        if not self.no_tf_pe:
            t = t + self.tf_pe.unsqueeze(1)
        if not self.no_pose_emb:
            t = t + self.pose_emb(pose_feat).view(1, self.nv, 1, self.C)
        return t, poses, pose_feat, fine

    def _decode(self, x_tok, fine_tok, poses):
        """Parent decoder, with the B3 global context applied as a zero-init FiLM on the decoder input."""
        if getattr(self, "ctx", None) is None:
            return super()._decode(x_tok, fine_tok, poses)
        B = x_tok.size(0); dev = x_tok.device
        x = x_tok.transpose(1, 2).reshape(B, self.C, self.lh, self.lw)
        gamma, beta = self.ctx_film(self.ctx).chunk(2, -1)
        x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        fine_erp = self._fine_lift(fine_tok, poses, dev)
        x = self.up_stages[0](x) + self.fine_to_dec(fine_erp)
        for st in self.up_stages[1:]:
            x = st(x)
        return torch.sigmoid(self.head(x))

    def afm_pretrained_param_names(self):
        return self.enc.afm.pretrained_param_names(prefix="enc.afm.")


def build_afm_model(args_dict, pretrained=True):
    """Construct OAAv2DepthAFM from a trainer/checkpoint args dict (mirrors core.ckpt.build's OAA kwargs)."""
    a = args_dict
    return OAAv2DepthAFM(
        audio_backbone=a["audio_backbone"], afm_pretrained=pretrained,
        afm_stem=a.get("afm_stem", "linear") or "linear",
        afm_input_norm=a.get("afm_input_norm", "std") or "std",
        C=a.get("dim", 256), nviews=a.get("nviews", 4), rounds=a.get("rounds", 2),
        lh=a.get("lift_h", 16), lw=a.get("lift_w", 32), dec_deep=a.get("dec_deep", True),
        stem_stride1=a.get("stem_stride1", False) or False, max_depth=a.get("max_depth", 10.0),
        no_pose_emb=a.get("no_pose_emb", False) or False, no_ray_emb=a.get("no_ray_emb", False) or False,
        no_geo_bias=a.get("no_geo_bias", False) or False, no_tf_pe=a.get("no_tf_pe", False) or False,
        no_cross=a.get("no_cross", False) or False,
        mic_diff=a.get("mic_diff", "none") or "none", fine_res=a.get("fine_res", False) or False,
        afm_patch=tuple(a["afm_patch"]) if a.get("afm_patch") else None)


def make_param_groups(model, base_lr, afm_lr_ratio=0.1, wd=1e-4, llrd=0.0):
    """AdamW groups: pretrained AFM tensors at afm_lr_ratio*base_lr, everything else at base_lr.
    llrd>0 adds layer-wise LR decay inside the pretrained group: block i (0=shallow) gets
    afm_lr * llrd^(depth-1-i); embeddings (pos/cls/pre_norm) get the shallowest (lowest) LR."""
    pre = model.afm_pretrained_param_names()
    afm_lr = base_lr * afm_lr_ratio
    depth = len(model.enc.afm.blocks)
    if not llrd:
        g_pre, g_new = [], []
        for n, p in model.named_parameters():
            (g_pre if n in pre else g_new).append(p)
        n_pre = sum(p.numel() for p in g_pre)
        print(f"[afm] LR groups: pretrained {len(g_pre)} tensors / {n_pre/1e6:.1f}M @ {afm_lr:.1e} | "
              f"base {len(g_new)} tensors / {sum(p.numel() for p in g_new)/1e6:.1f}M @ {base_lr:.1e}", flush=True)
        return [{"params": g_pre, "lr": afm_lr, "weight_decay": wd},
                {"params": g_new, "lr": base_lr, "weight_decay": wd}]
    import re as _re
    buckets = {i: [] for i in range(-1, depth)}                            # -1 = embeddings
    g_new = []
    for n, p in model.named_parameters():
        if n not in pre:
            g_new.append(p); continue
        m = _re.search(r"blocks\.(\d+)\.", n)
        buckets[int(m.group(1)) if m else (depth - 1 if "out_norm" in n else -1)].append(p)
    groups = [{"params": g_new, "lr": base_lr, "weight_decay": wd}]
    for i in range(-1, depth):
        if buckets[i]:
            lr_i = afm_lr * (llrd ** (depth - 1 - max(i, 0)))
            groups.append({"params": buckets[i], "lr": lr_i, "weight_decay": wd})
    print(f"[afm] LLRD groups: base {base_lr:.1e} | afm block LRs "
          f"{afm_lr*(llrd**(depth-1)):.1e} (emb/blk0) -> {afm_lr:.1e} (blk{depth-1}); decay {llrd}", flush=True)
    return groups
