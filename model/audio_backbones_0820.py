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

AFM_DIR = os.environ.get("AFM_WEIGHTS", "/root/local1/changwoo/_afm_weights")
BACKBONES = ("cnn", "audiomosaic", "bat", "eat", "sslam", "m2d", "m2d_plain", "m2d20ms")

# style: block forward order; layout: pos-embed native grid axis order ("tf" rows=time / "ft" rows=freq)
_SPECS = {
    "audiomosaic": dict(hf="hanxunh/AudioMosaic-vit-b16-pretrained", prefix="", style="prenorm",
                        grid=(64, 8), layout="tf", pos_key="pos_embed", pos_has_cls=True,
                        cls_key="cls_token", pre_norm_key="norm_pre", final_norm_key=None, cls_gets_pos=True),
    "bat": dict(hf="lrauch/BAT-vit-b16-pretrainedAS2M", prefix="", style="postnorm_gate",
                grid=(64, 8), layout="tf", pos_key="pos_embed", pos_has_cls=False,
                cls_key="cls_token", pre_norm_key="pre_norm", final_norm_key=None, cls_gets_pos=False),
    "eat": dict(hf="worstchan/EAT-base_epoch30_pretrain", prefix="model.", style="postnorm_alt",
                grid=(64, 8), layout="tf", pos_key="fixed_positional_encoder.positions", pos_has_cls=False,
                cls_key="extra_tokens", pre_norm_key="pre_norm", final_norm_key=None, cls_gets_pos=False),
    "sslam": dict(hf="ta012/SSLAM_pretrain", prefix="model.", style="postnorm_alt",
                  grid=(64, 8), layout="tf", pos_key="fixed_positional_encoder.positions", pos_has_cls=False,
                  cls_key="extra_tokens", pre_norm_key="pre_norm", final_norm_key=None, cls_gets_pos=False),
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


class AFMBackbone(nn.Module):
    """Pretrained ViT-B/16 audio encoder with a task-specific patch embedding.
    forward(x: [B*, 1, 256, 512] magnitude spec) -> [B*, lh*lw, out_dim] patch tokens (no CLS/no pooling)."""
    DIM, DEPTH = 768, 12

    def __init__(self, name, out_dim=256, lh=LH, lw=LW, pretrained=True, verbose=True, stem="linear",
                 input_norm="std"):
        super().__init__()
        assert name in _SPECS, f"unknown audio backbone {name} (choose from {list(_SPECS)})"
        assert stem in ("linear", "conv"), f"bad afm stem {stem}"
        # AFM input statistics: "std" = log1p + per-sample standardize (default);
        # "db" = 20*log10 (AudioSet log-mel-like) + per-sample standardize;
        # "db_minmax" = dB + per-sample min-max to [0,1] (BAT's native per_sample_minmax_after_db).
        assert input_norm in ("std", "db", "db_minmax"), f"bad afm input norm {input_norm}"
        self.input_norm = input_norm
        s = _SPECS[name]
        self.name, self.lh, self.lw, self.stem_kind = name, lh, lw, stem
        self.cls_gets_pos = s["cls_gets_pos"]
        ph, pw = s.get("patch", (256 // lh, 512 // lw))                   # default 16x16 on the 256x512 spec
        self.grid_hw = (256 // ph, 512 // pw)                             # token grid (freq rows, time cols)
        M = self.grid_hw[0] * self.grid_hw[1]
        assert M == lh * lw, f"patch {ph}x{pw} gives {M} tokens, OAA needs {lh * lw}"
        if stem == "conv":
            # conv stem variant ("early convolutions help transformers"): 4 x stride-2 3x3 convs -> same
            # 16x32 grid but sub-patch locality preserved. NEW params (base LR); linear patch is the default.
            assert (ph, pw) == (16, 16), "conv stem only supports the 16x16 grid"
            ch = [1, 64, 128, 256, self.DIM]
            layers = []
            for i in range(4):
                layers += [nn.Conv2d(ch[i], ch[i + 1], 3, 2, 1), nn.GELU()]
            self.patch = nn.Sequential(*layers[:-1])                      # drop trailing GELU (linear-out like ViT stem)
        else:
            self.patch = nn.Conv2d(1, self.DIM, (ph, pw), (ph, pw))       # NEW (task-specific, base LR)
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
                  f"style={s['style']}; new params: patch-embed, proj"
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
        return set(names)


# --------------------------------------------------------------------------- OAA integration
class AFMViewEncoder(nn.Module):
    """Drop-in replacement for model.oaa.ViewEncoder: AFM coarse tokens + the ORIGINAL lightweight
    CNN fine path (ViewEncoder truncated at its (2lh, 2lw) fine tap, weights fresh, base LR)."""
    def __init__(self, name, C=256, ngf=64, in_ch=1, norm="group", lh=LH, lw=LW,
                 enc_res=(256, 512), stem_stride1=False, pretrained=True, afm_stem="linear",
                 afm_input_norm="std"):
        super().__init__()
        assert in_ch == 1, "AFM encoder supports 1 channel per observation"
        fe = ViewEncoder(C, ngf, in_ch, norm, lh, lw, enc_res, stem_stride1)
        stages = int(round(math.log2(enc_res[0] / lh)))
        fe.net = fe.net[: 2 * (stages - 1)]                               # keep up to the (2lh,2lw) fine tap
        self.fine_enc = fe
        self.fine_ch = fe.fine_ch
        self.afm = AFMBackbone(name, out_dim=C, lh=lh, lw=lw, pretrained=pretrained, stem=afm_stem,
                               input_norm=afm_input_norm)
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
    Everything downstream of the encoder is inherited unchanged."""
    def __init__(self, audio_backbone, afm_pretrained=True, afm_stem="linear", afm_input_norm="std", **kw):
        super().__init__(**kw)
        assert audio_backbone in _SPECS, f"bad audio_backbone {audio_backbone}"
        self.audio_backbone = audio_backbone
        self.enc = AFMViewEncoder(audio_backbone, C=self.C, in_ch=self.in_ch, lh=self.lh, lw=self.lw,
                                  enc_res=self.enc_res, stem_stride1=kw.get("stem_stride1", False),
                                  pretrained=afm_pretrained, afm_stem=afm_stem, afm_input_norm=afm_input_norm)
        # fine_in built by super() from the full ViewEncoder's fine_ch; the truncated fine path keeps
        # the same channel count by construction — assert instead of trusting it silently.
        assert self.fine_in.in_features == self.enc.fine_ch, \
            f"fine_ch mismatch {self.fine_in.in_features} vs {self.enc.fine_ch}"

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
        no_cross=a.get("no_cross", False) or False)


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
