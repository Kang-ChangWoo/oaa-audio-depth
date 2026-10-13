"""OAA: Orientation-Aligned Alternating attention for multi-observation binaural audio -> ERP depth.

Input contract: spec (B, nviews, H, W) magnitude spectrograms, one channel per posed ear-specific
observation. Each observation is encoded INDEPENDENTLY by a weight-shared encoder (batched over
the batch dim, not channel-stacked), conditioned on its time-frequency position and its pose
(receiver yaw + ear identity). Observations are fused by alternating intra-/inter-observation
attention, then 16x32 ray-conditioned panoramic queries cross-attend the fused tokens with a
geometry-aware bias, are refined by self-attention, and decoded to a 256x512 depth map.

Coordinate convention:
  ERP azimuth A in [-pi, pi); ray d(A) = (sinA, 0, cosA): A=0 -> +z (front), A=+pi/2 -> +x.
  A receiver at yaw psi faces d(psi); a global ray seen in that receiver's frame is R(-psi) d.
Observation pose list per input mode (yaw, ear): ear -1 = left, +1 = right.

Ablation switches (construction-time, default off => reference model):
  no_tf_pe, no_pose_emb, no_ray_emb, no_geo_bias, no_cross.

Attention/fusion variants (construction-time, default off => reference model, and the extra
parameters are only *created* when the switch is on, so an original checkpoint still loads with
strict=True):
  qk_norm      RayMicAttn RMS-normalised Q/K + learnable per-head log-scales on content and bias
  ray_gate     RayMicAttn identity-initialised per-(ray, head) output gate (2*sigmoid, =1 at init)
  view_fusion  "global" (reference single softmax) | "hier" (exact two-stage decomposition of the
               same softmax, no new parameters) | "hier-geo" (hier + zero-initialised geometry
               routing residual on the outer logits, so it starts exactly at "hier")
Optional runtime argument (no parameters, no effect when omitted):
  forward(..., obs_valid=(B,N) bool)   explicit observation-validity masking of the inter-observation
               keys, the ray-cross keys/values and the fine-lift mean.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LH, LW = 16, 32                      # panoramic query grid (elevation x azimuth)

VIEW_FUSIONS = ("global", "hier", "hier-geo")
_QK_LOGSCALE_CLIP = 4.0              # exp() multiplier bounded to [1.8e-2, 5.5e1]
_QK_EPS = 1e-6
# Tests flip this to False to exercise the masked code path on all-valid input; in production the
# short-circuit is what makes "obs_valid all True" bit-identical to "obs_valid=None".
_SHORTCIRCUIT_ALL_VALID = True


def _mask_engaged(obs_valid):
    """True when the explicit-validity code path must run for this batch."""
    if obs_valid is None:
        return False
    return (not _SHORTCIRCUIT_ALL_VALID) or (not bool(obs_valid.all()))


def _hiprec(x):
    """Promote reduced precision to float32 for softmax / logsumexp, and leave float32 or float64
    alone. Plain `.float()` would *downcast* a float64 tensor and silently cap the precision of the
    equivalence tests at 1e-7."""
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def _rms_norm_heads(x, eps=_QK_EPS):
    """RMS-normalise over the last (head) dim, computed in float32 for numerical stability.

    Note on the scaling convention: an RMS-normalised vector has L2 norm sqrt(dk), so
    `rms(Q) @ rms(K)^T / sqrt(dk)` is algebraically identical to `l2(Q) @ l2(K)^T * sqrt(dk)`.
    A separate `--qk-norm-type l2` switch would therefore only move the starting point of the
    learnable log-scale, which is why only the RMS variant is implemented.
    """
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)

# pose (yaw, ear_sign) of each channel for the fixed input modes
_ALL_POSES = [(0.0, -1.0), (0.0, 1.0), (math.pi / 2, 1.0), (-math.pi / 2, -1.0)]   # [0L, 0R, 90R, 270L]
_POOL8 = [(0.0, -1.0), (0.0, 1.0), (math.pi / 2, -1.0), (math.pi / 2, 1.0),
          (math.pi, -1.0), (math.pi, 1.0), (-math.pi / 2, -1.0), (-math.pi / 2, 1.0)]   # [0L,0R,90L,90R,180L,180R,270L,270R]


def _dir_pe(h, w, device):
    """Unit direction per ERP cell -> [dx,dy,dz,dx^2,dy^2,dz^2]. d(A=0)=+z (front), d(+pi/2)=+x."""
    el = (torch.arange(h, device=device) + 0.5) / h * math.pi - math.pi / 2
    az = (torch.arange(w, device=device) + 0.5) / w * 2 * math.pi - math.pi
    E, A = torch.meshgrid(el, az, indexing="ij")
    dx = torch.cos(E) * torch.sin(A); dy = torch.sin(E); dz = torch.cos(E) * torch.cos(A)
    d = torch.stack([dx, dy, dz, dx * dx, dy * dy, dz * dz], -1)
    return d.reshape(h * w, 6)


def _yaw_rot_inv(dir3, yaw):
    """R(-yaw) applied to ray dirs (...,3): global ray -> receiver-local frame.  d(A) -> d(A - yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy, dz = dir3[..., 0], dir3[..., 1], dir3[..., 2]
    return torch.stack([dx * c - dz * s, dy, dx * s + dz * c], -1)


def _norm(co, kind):
    return nn.GroupNorm(8, co) if kind == "group" else nn.BatchNorm2d(co)


class _ResBlk(nn.Module):
    def __init__(self, ci, co, down=True, norm="group"):
        super().__init__()
        s = 2 if down else 1
        self.c1 = nn.Conv2d(ci, co, 3, s, 1); self.n1 = _norm(co, norm)
        self.c2 = nn.Conv2d(co, co, 3, 1, 1); self.n2 = _norm(co, norm)
        self.sc = nn.Conv2d(ci, co, 1, s, 0) if (ci != co or down) else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x))); h = self.n2(self.c2(h))
        return F.gelu(h + self.sc(x))


class ViewEncoder(nn.Module):
    """Shared per-observation encoder: strided conv stem + residual downsampling to (lh, lw) tokens.
    Also taps the (2lh, 2lw) feature map ("fine" tokens) for the multi-scale decoder skip."""
    def __init__(self, C=256, ngf=64, in_ch=1, norm="group", lh=LH, lw=LW, enc_res=(256, 512),
                 stem_stride1=False):
        super().__init__()
        self.enc_res = enc_res; self.stem_stride1 = stem_stride1
        # stem_stride1: stride-1 stem on the native-resolution input instead of 2x-upsample + stride-2
        # (mathematically equivalent receptive field, no interpolation, 1/4 compute)
        self.stem = nn.Sequential(nn.Conv2d(in_ch, ngf // 2, 3, (1 if stem_stride1 else 2), 1), nn.GELU(),
                                  nn.Conv2d(ngf // 2, ngf, 3, 1, 1))
        stages = int(round(math.log2(enc_res[0] / lh)))                             # post-stem downsamplings
        assert enc_res[0] // lh == enc_res[1] // lw == 2 ** stages, f"enc_res {enc_res} -> {lh}x{lw}"
        assert stages >= 2, "multi-scale decoder needs >=2 encoder stages (fine tap at (2lh,2lw))"
        chans = [ngf, ngf * 2, ngf * 4, C, C, C][:stages] + [C]
        blocks = []
        for i in range(stages):
            blocks += [_ResBlk(chans[i], chans[i], down=False, norm=norm), _ResBlk(chans[i], chans[i + 1], norm=norm)]
        self.net = nn.Sequential(*blocks); self.C = C
        self.lh, self.lw = lh, lw
        self.fine_ch = chans[stages - 1]                                             # channels at the (2lh,2lw) tap

    def forward(self, x):
        want = tuple(self.enc_res) if self.stem_stride1 else tuple(2 * r for r in self.enc_res)
        if x.shape[-2:] != want:
            x = F.interpolate(x, size=want, mode="bilinear", align_corners=False)
        h = self.stem(x); ft = None
        for blk in self.net:
            h = blk(h)
            if h.shape[-2:] == (2 * self.lh, 2 * self.lw):                           # fine tap
                ft = h.flatten(2).transpose(1, 2)                                    # (B, 4*lh*lw, fine_ch)
        tok = h.flatten(2).transpose(1, 2)                                           # (B, lh*lw, C)
        return tok, ft


class SelfAttn(nn.Module):
    """Pre-LN self-attention block (query refinement)."""
    def __init__(self, C, heads=8):
        super().__init__()
        self.n = nn.LayerNorm(C); self.a = nn.MultiheadAttention(C, heads, batch_first=True)
        self.m = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))

    def forward(self, x):
        h = self.n(x)
        x = x + self.a(h, h, h, need_weights=False)[0]
        return x + self.m(x)


class CondSelfAttn(nn.Module):
    """Intra-observation attention with AdaLN pose conditioning. The modulation layer is zero-initialised,
    so the block starts as a plain (pose-independent) transformer block."""
    def __init__(self, C, heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(C, elementwise_affine=False)
        self.a = nn.MultiheadAttention(C, heads, batch_first=True)
        self.n2 = nn.LayerNorm(C, elementwise_affine=False)
        self.m = nn.Sequential(nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))
        self.ada = nn.Linear(C, 6 * C)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, cond):                      # x: (B*, M, C)  cond: (B*, C) per-observation pose emb
        sa, ba, ga, sm, bm, gm = self.ada(cond).unsqueeze(1).chunk(6, -1)
        h = self.n1(x) * (1 + sa) + ba
        x = x + ga * self.a(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + sm) + bm
        return x + gm * self.m(h)


class InterMicAttn(nn.Module):
    """Inter-observation attention: each token attends the same token position across the N observations."""
    def __init__(self, C, heads=8):
        super().__init__()
        self.n = nn.LayerNorm(C); self.a = nn.MultiheadAttention(C, heads, batch_first=True)
        self.m = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))

    def forward(self, Fm, obs_valid=None):     # (B, N, M, C)
        B, N, M, C = Fm.shape
        x = Fm.permute(0, 2, 1, 3).reshape(B * M, N, C)
        kpm = None
        if _mask_engaged(obs_valid):
            # query batch is (B*M, N) with index b*M + m, so repeat the per-observation flag over M
            kpm = (~obs_valid).unsqueeze(1).expand(B, M, N).reshape(B * M, N)
        o, _ = self.a(*[self.n(x)] * 3, need_weights=False, key_padding_mask=kpm)
        Fm = Fm + o.reshape(B, M, N, C).permute(0, 2, 1, 3)
        return Fm + self.m(Fm)


class RayMicAttn(nn.Module):
    """Geometry-aware cross-attention: panoramic ray queries attend all observation tokens. A per-(ray,
    observation) bias from [R_i^T r_j (3), ray . ear_axis (1), ear_sign (1)] is added to the logits before
    softmax. LN + residual + FFN."""
    def __init__(self, C, heads=8, use_bias=True, qk_norm=False, ray_gate=False,
                 view_fusion="global"):
        super().__init__()
        assert view_fusion in VIEW_FUSIONS, f"view_fusion {view_fusion} not in {VIEW_FUSIONS}"
        self.h, self.dk = heads, C // heads
        self.use_bias = use_bias          # False = "w/o geometric bias" ablation (content-only QK)
        self.nq = nn.LayerNorm(C); self.nk = nn.LayerNorm(C)
        self.q = nn.Linear(C, C); self.k = nn.Linear(C, C); self.v = nn.Linear(C, C); self.o = nn.Linear(C, C)
        self.bias_mlp = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, heads))
        self.ffn = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))
        # ---- variants. Each one declares parameters ONLY when it is on, so the default model's
        # state_dict is byte-for-byte the reference one and old checkpoints load with strict=True.
        self.qk_norm, self.ray_gate, self.view_fusion = qk_norm, ray_gate, view_fusion
        if qk_norm:
            self.logit_scale = nn.Parameter(torch.zeros(heads))   # log-scale, init 0 -> multiplier 1
            self.geo_scale = nn.Parameter(torch.zeros(heads))
        if ray_gate:
            self.gate_proj = nn.Linear(C, heads)                  # 2*sigmoid(0) = 1 -> identity at init
            nn.init.zeros_(self.gate_proj.weight); nn.init.zeros_(self.gate_proj.bias)
        if view_fusion == "hier-geo":
            self.route_mlp = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, heads))
            nn.init.zeros_(self.route_mlp[-1].weight); nn.init.zeros_(self.route_mlp[-1].bias)
        self.collect_stats = False        # diagnostics only; never read by the loss
        self.stats = {}

    def _geo_feats(self, ray_dir3, poses):
        """Per-observation geometry features [R(-yaw) r (3), r.ear_axis * ear (1), ear (1)].

        Returned as the per-observation list the reference bias path already consumed, so that
        `bias_mlp` sees exactly the same call shapes as before and the routing MLP can reuse the
        features without recomputing them.
        """
        feats = []
        for yaw, ear in poses:
            local = _yaw_rot_inv(ray_dir3, yaw)
            # dtype follows ray_dir3 so the module also runs in float64; at float32 (and under bf16
            # autocast, where ray_dir3 is the float32 dir buffer) this is the original float32 tensor
            a = torch.tensor([math.cos(yaw), 0.0, -math.sin(yaw)],
                             device=ray_dir3.device, dtype=ray_dir3.dtype)   # ear axis R(yaw)@x_hat
            c = (ray_dir3 @ a).unsqueeze(-1) * ear
            e = torch.full_like(c, float(ear))
            feats.append(torch.cat([local, c, e], -1))                     # (R,5)
        return feats

    def _logscale(self, p):
        return p.clamp(-_QK_LOGSCALE_CLIP, _QK_LOGSCALE_CLIP).exp()

    def _hier_fuse(self, logits, V, N, M, feats, kv_keep):
        """Exact two-stage decomposition of the single global softmax over the N*M acoustic tokens:

            p(t|i,ray) = softmax_t L_{i,t};   z_i = sum_t p(t|i,ray) V_{i,t}
            logZ_i     = logsumexp_t L_{i,t}; alpha_i = softmax_i logZ_i
            y          = sum_i alpha_i z_i            ==  softmax over ALL (i,t) applied to V

        A plain mean over observations is NOT this and is deliberately not implemented.
        The reference per-observation geometry bias is constant across the M tokens of an
        observation, so it shifts logsumexp_t by exactly that constant: geometry enters logZ once,
        it is not double-counted.

        'hier-geo' adds a zero-initialised learned residual to logZ (content mass is kept, not
        replaced), so at initialisation the output equals 'hier' exactly.
        """
        B, h, R, _ = logits.shape
        L = _hiprec(logits.reshape(B, h, R, N, M))
        Vn = _hiprec(V.reshape(B, h, N, M, self.dk))
        # within-observation weights come from the UNMASKED logits so that a fully dropped
        # observation yields a finite (merely unused) z_i instead of softmax(-inf) = NaN
        within = L.softmax(-1)                                             # B,h,R,N,M
        per_view = torch.einsum("bhrnm,bhnmd->bhrnd", within, Vn)          # B,h,R,N,dk
        Lz = L if kv_keep is None else L.masked_fill(~kv_keep.reshape(B, 1, 1, N, M), float("-inf"))
        logZ = torch.logsumexp(Lz, dim=-1)                                 # B,h,R,N (-inf if dropped)
        delta = None
        if self.view_fusion == "hier-geo":
            delta = self.route_mlp(torch.stack(feats, 1))                  # (R,N,h)
            logZ = logZ + _hiprec(delta.permute(2, 0, 1).unsqueeze(0))      # -inf + finite stays -inf
        alpha = logZ.softmax(-1)                                           # dropped observation -> exactly 0
        out = torch.einsum("bhrn,bhrnd->bhrd", alpha, per_view)
        if self.collect_stats:
            with torch.no_grad():
                # separate the axes so a change in alpha can be attributed: per view, and per (head, view)
                self.stats["alpha_mean_per_view"] = alpha.mean((0, 1, 2)).detach().float().cpu()
                self.stats["alpha_mean_head_view"] = alpha.mean((0, 2)).detach().float().cpu()
                self.stats["alpha_entropy"] = (-(alpha.clamp_min(1e-9).log() * alpha)).sum(-1).mean().item()
                if delta is not None:
                    self.stats["route_delta_absmean"] = delta.abs().mean().item()
                    self.stats["route_delta_per_view"] = delta.mean((0, 2)).detach().float().cpu()
        return out.to(V.dtype)

    def forward(self, q_in, tokens, ray_dir3, poses, M, obs_valid=None):
        B, R, C = q_in.shape; N = len(poses)
        q_ln = self.nq(q_in)
        Q = self.q(q_ln).view(B, R, self.h, self.dk).transpose(1, 2)
        tk = self.nk(tokens)
        K = self.k(tk).view(B, -1, self.h, self.dk).transpose(1, 2)
        V = self.v(tk).view(B, -1, self.h, self.dk).transpose(1, 2)
        feats = self._geo_feats(ray_dir3, poses) if (self.use_bias or self.view_fusion == "hier-geo") else None
        if self.qk_norm:
            # decouple the content logit scale from ||Q||,||K|| so the geometry bias and the content
            # term keep a learnable, head-wise relative size instead of an accidental one
            content = (_rms_norm_heads(Q) @ _rms_norm_heads(K).transpose(-2, -1)) / math.sqrt(self.dk)
            logits = (content * self._logscale(self.logit_scale).view(1, -1, 1, 1)).to(Q.dtype)
        else:
            logits = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)
        if self.use_bias:
            bmic = torch.stack([self.bias_mlp(f) for f in feats], 1)       # (R,N,h)
            if self.qk_norm:
                bmic = bmic * self._logscale(self.geo_scale).view(1, 1, -1)
            bfull = bmic.unsqueeze(2).expand(R, N, M, self.h).reshape(R, N * M, self.h)
            logits = logits + bfull.permute(2, 0, 1).unsqueeze(0)
        kv_keep = None
        if _mask_engaged(obs_valid):
            kv_keep = obs_valid[:, :, None].expand(B, N, M).reshape(B, 1, 1, N * M)
        if self.view_fusion == "global":
            if kv_keep is None:
                out_head = logits.softmax(-1) @ V                          # reference path, unchanged
            else:                                                          # FP32 softmax; dropped keys get weight 0
                out_head = _hiprec(logits).masked_fill(~kv_keep, float("-inf")).softmax(-1).to(V.dtype) @ V
        else:
            out_head = self._hier_fuse(logits, V, N, M, feats, kv_keep)
        if self.ray_gate:
            gate = 2.0 * torch.sigmoid(self.gate_proj(q_ln))               # (B,R,h) in (0,2), =1 at init
            out_head = out_head * gate.permute(0, 2, 1).unsqueeze(-1)
            if self.collect_stats:
                with torch.no_grad():
                    g = gate.detach().float()
                    self.stats.update(gate_mean=g.mean().item(), gate_std=g.std().item(),
                                      gate_min=g.min().item(), gate_max=g.max().item(),
                                      gate_frac_lt01=(g < 0.1).float().mean().item(),
                                      gate_frac_gt19=(g > 1.9).float().mean().item())
                    self.stats["gate_mean_per_head"] = g.mean((0, 1)).cpu()
        if self.collect_stats:
            with torch.no_grad():
                lf = logits.detach().float()
                self.stats["logit_rms_per_head"] = lf.pow(2).mean((0, 2, 3)).sqrt().cpu()
                p = lf.softmax(-1)
                self.stats["attn_entropy"] = (-(p.clamp_min(1e-9).log() * p)).sum(-1).mean().item()
                if self.use_bias:
                    self.stats["geo_bias_rms_per_head"] = bmic.detach().float().pow(2).mean((0, 1)).sqrt().cpu()
                if self.qk_norm:
                    self.stats["logit_scale"] = self._logscale(self.logit_scale).detach().float().cpu()
                    self.stats["geo_scale"] = self._logscale(self.geo_scale).detach().float().cpu()
        out = out_head.transpose(1, 2).reshape(B, R, C)
        h = q_in + self.o(out)
        return h + self.ffn(h)


def _make_up(C, norm, lh, lw, deep=True):
    """Decoder stages (lh,lw) -> (256,512), #stages = log2(256/lh); returns (ModuleList, per-stage out channels)."""
    stages = int(round(math.log2(256 / lh)))
    assert 256 % lh == 0 and 512 % lw == 0 and (256 // lh) == (512 // lw), f"bad grid {lh}x{lw}"
    chans = [C, 128, 64, 32, 16, 16, 16][:stages + 1]

    def up(ci, co):
        blks = [nn.ConvTranspose2d(ci, co, 4, 2, 1), _norm(co, norm), nn.GELU(), _ResBlk(co, co, down=False, norm=norm)]
        if deep: blks.append(_ResBlk(co, co, down=False, norm=norm))
        return nn.Sequential(*blks)
    return nn.ModuleList([up(chans[i], chans[i + 1]) for i in range(stages)]), chans[1:stages + 1]


class OAAv2Depth(nn.Module):
    """Reference OAA model (the one used for all reported results)."""
    def __init__(self, C=256, rounds=2, in_ch=1, norm="group", lh=LH, lw=LW, dec_deep=True,
                 enc_res=(256, 512), nviews=4, stem_stride1=False, max_depth=10.0,
                 no_pose_emb=False, no_ray_emb=False, no_geo_bias=False, no_tf_pe=False, no_cross=False,
                 qk_norm=False, ray_gate=False, view_fusion="global"):
        super().__init__()
        # the attention/fusion variants all live inside RayMicAttn, which --no-cross removes entirely
        if no_cross:
            assert not (qk_norm or ray_gate or view_fusion != "global"), (
                "--no-cross removes RayMicAttn, so --qk-norm / --ray-gate / --view-fusion have no "
                "effect; refusing the combination instead of silently ignoring it")
        self.qk_norm, self.ray_gate, self.view_fusion = qk_norm, ray_gate, view_fusion
        # ablations: pathways removed at construction; defaults keep the reference behaviour
        self.no_pose_emb = no_pose_emb    # heading/ear embedding not added to tokens (+AdaLN cond zeroed)
        self.no_ray_emb = no_ray_emb      # learnable queries only (dir_mlp / glob_dir / fine_dir_mlp off)
        self.no_geo_bias = no_geo_bias    # RayMicAttn geometric bias off
        self.no_tf_pe = no_tf_pe          # coarse-token TF positional encoding off (NOTE: the fine-skip path keeps fine_tf_pe,
                                          #   and no_geo_bias keeps the deterministic yaw roll of the fine skip — as in the reported runs)
        self.no_cross = no_cross          # ray-observation cross-attention replaced by observation mean + reshape
        self.C = C; self.in_ch = in_ch; self.nv = nviews
        self.lh, self.lw, self.M = lh, lw, lh * lw; self.enc_res = enc_res
        self.max_depth_n = max_depth
        self.enc = ViewEncoder(C, in_ch=in_ch, norm=norm, lh=lh, lw=lw, enc_res=enc_res, stem_stride1=stem_stride1)
        self.pose_emb = nn.Sequential(nn.Linear(3, C), nn.GELU(), nn.Linear(C, C))       # MLP_pose([sin, cos, ear])
        self.tf_pe = nn.Parameter(torch.zeros(1, self.M, C))                              # learnable TF PE
        self.q = nn.Parameter(torch.randn(1, self.M, C) * 0.02)                           # learnable panoramic queries
        self.dir_mlp = nn.Sequential(nn.Linear(6, C), nn.GELU(), nn.Linear(C, C))        # MLP_ray (query side)
        self.glob_dir = nn.Sequential(nn.Linear(6, C), nn.GELU(), nn.Linear(C, C))       # ray emb added after fusion
        self.erp = nn.ModuleList([SelfAttn(C) for _ in range(4)])                         # 4 refinement blocks
        self.aux_head = nn.Linear(C, 1)   # coarse-depth head; kept for checkpoint compatibility (not used by the loss)
        if nviews == 8:                                                  # loader mode 'r8' channel order
            self.view_pose = list(_POOL8)
        elif nviews == 6:                                                # loader mode 'r6' order: 0LR + 90LR + 270LR
            self.view_pose = [_POOL8[j] for j in (0, 1, 2, 3, 6, 7)]
        elif nviews <= 4:
            self.view_pose = _ALL_POSES[:nviews]                        # nviews=2 -> [0L,0R]; 4 -> [0L,0R,90R,270L]
        else:
            # microphone-scaling study: more binaural headings than the released rig has. Only a
            # placeholder -- the loader passes the true view_poses at forward time (data_micgain
            # records the dataset's own bisection ordering) and _pose_tensors prefers those.
            assert nviews % 2 == 0, f"nviews {nviews} must be even (binaural headings)"
            k = nviews // 2
            self.view_pose = [(2 * math.pi * (j // 2) / k, (-1.0, 1.0)[j % 2]) for j in range(nviews)]
        self.register_buffer("dir6_buf", _dir_pe(lh, lw, torch.device("cpu")), persistent=False)
        # multi-scale decoder: stage-0 upsample (lh,lw)->(2lh,2lw) + fine ray-lift skip, then remaining stages
        self.up_stages, up_ch = _make_up(C, norm, lh, lw, deep=dec_deep)
        self.fine_in = nn.Linear(self.enc.fine_ch, C)
        self.fine_q = nn.Parameter(torch.randn(1, 4 * self.M, C) * 0.02)
        self.fine_tf_pe = nn.Parameter(torch.zeros(1, 4 * self.M, C))
        self.fine_dir_mlp = nn.Sequential(nn.Linear(6, C), nn.GELU(), nn.Linear(C, C))
        self.fine_ln = nn.LayerNorm(C)
        self.fine_lift = nn.MultiheadAttention(C, 8, batch_first=True)
        self.fine_to_dec = nn.Conv2d(C, up_ch[0], 1)                    # zero-init: starts with no injection
        nn.init.zeros_(self.fine_to_dec.weight); nn.init.zeros_(self.fine_to_dec.bias)
        self.register_buffer("fine_dir6_buf", _dir_pe(2 * lh, 2 * lw, torch.device("cpu")), persistent=False)
        self.head = nn.Conv2d(up_ch[-1], 1, 3, padding=1)
        # fusion
        self.intra = nn.ModuleList([CondSelfAttn(C) for _ in range(rounds)])
        self.inter = nn.ModuleList([InterMicAttn(C) for _ in range(rounds)])
        self.ray_mic = RayMicAttn(C, use_bias=not no_geo_bias, qk_norm=qk_norm,
                                  ray_gate=ray_gate, view_fusion=view_fusion)

    # ------------------------------------------------------------------ helpers
    def _pose_tensors(self, dev, view_pose=None):
        """-> pose_feat (nv,3) = [sin yaw, cos yaw, ear], poses list[(yaw, ear)]."""
        vp = view_pose if view_pose is not None else self.view_pose
        # dtype follows pose_emb's weight (float32 in every released run, float64 only in tests)
        pose_feat = torch.tensor([[math.sin(y), math.cos(y), e] for y, e in vp],
                                 device=dev, dtype=self.pose_emb[0].weight.dtype)
        return pose_feat, [tuple(p) for p in vp]

    def _encode(self, spec, view_pose=None):
        """Batched per-observation encode (NOT channel-stacked) + TF PE + pose emb.
        -> tokens (B,nv,M,C), poses, fine tokens (B,nv,4M,fine_ch)."""
        assert view_pose is None or len(view_pose) == self.nv, f"view_pose len must be {self.nv}"
        assert spec.size(1) == self.nv * self.in_ch, f"expected {self.nv * self.in_ch}ch, got {spec.size(1)}"
        B = spec.size(0); dev = spec.device; H, W = spec.shape[-2:]
        pose_feat, poses = self._pose_tensors(dev, view_pose)
        v = spec.view(B, self.nv, self.in_ch, H, W).reshape(B * self.nv, self.in_ch, H, W)
        enc_t, fine_t = self.enc(v)
        t = enc_t.reshape(B, self.nv, self.M, self.C)
        if not self.no_tf_pe:
            t = t + self.tf_pe.unsqueeze(1)
        fine = fine_t.reshape(B, self.nv, 4 * self.M, self.enc.fine_ch)
        if not self.no_pose_emb:
            t = t + self.pose_emb(pose_feat).view(1, self.nv, 1, self.C)
        return t, poses, pose_feat, fine

    def _fine_lift(self, fine_tok, poses, dev, obs_valid=None):
        """Per-observation ray-lift of the fine (2lh,2lw) tokens -> yaw roll-align -> mean. -> (B,C,2lh,2lw).
        With obs_valid the mean becomes a masked mean, so a dropped observation's zeroed-spectrogram
        feature stops leaking into the decoder skip."""
        B, nv, R2, _ = fine_tok.shape
        d6 = self.fine_dir6_buf.to(dev)                                   # (R2,6)
        q = ((self.fine_q if self.no_ray_emb else
              self.fine_q + self.fine_dir_mlp(d6).unsqueeze(0))).expand(B * nv, -1, -1)
        kv = self.fine_in(fine_tok).reshape(B * nv, R2, self.C) + self.fine_tf_pe
        s = self.fine_lift(self.fine_ln(q), kv, kv, need_weights=False)[0].reshape(B, nv, R2, self.C)
        outs = []
        for i, (yaw, ear) in enumerate(poses):
            sh = int(round(yaw / (2 * math.pi) * (2 * self.lw)))          # align observation frame -> global azimuth
            si = s[:, i].transpose(1, 2).reshape(B, self.C, 2 * self.lh, 2 * self.lw)
            outs.append(torch.roll(si, shifts=sh, dims=-1))
        st = torch.stack(outs, 1)
        if not _mask_engaged(obs_valid):
            return st.mean(1)
        w = obs_valid[:, :, None, None, None].to(st.dtype)
        return (st * w).sum(1) / w.sum(1).clamp_min(1)

    def _decode(self, x_tok, fine_tok, poses, obs_valid=None):
        B = x_tok.size(0); dev = x_tok.device
        x = x_tok.transpose(1, 2).reshape(B, self.C, self.lh, self.lw)
        fine_erp = self._fine_lift(fine_tok, poses, dev, obs_valid)       # (B,C,2lh,2lw)
        x = self.up_stages[0](x) + self.fine_to_dec(fine_erp)
        for st in self.up_stages[1:]:
            x = st(x)
        return torch.sigmoid(self.head(x))                                # depth / max_depth in [0,1]

    # ------------------------------------------------------------------ forward
    def forward(self, spec, view_poses=None, obs_valid=None):
        """spec (B, nviews, H, W) magnitude spectrograms; view_poses optional list[(yaw, ear)] of length
        nviews overriding the fixed mode poses. obs_valid optional (B, nviews) bool (True = observed):
        when given, invalid observations are removed from the inter-observation keys, the ray-cross
        keys/values and the fine-skip mean. obs_valid=None reproduces the reference model exactly.
        Returns depth in [0,1] (multiply by max_depth for metres)."""
        B = spec.size(0); dev = spec.device; dir6 = self.dir6_buf.to(dev); dir3 = dir6[:, :3]
        if obs_valid is not None:
            assert obs_valid.shape == (B, self.nv), f"obs_valid must be ({B},{self.nv}), got {tuple(obs_valid.shape)}"
            obs_valid = obs_valid.to(dev, torch.bool)
            assert bool(obs_valid.any(1).all()), "every sample needs at least one valid observation"
        F4, poses, pose_feat, fine = self._encode(spec, view_poses)       # (B,nv,M,C)
        cond = self.pose_emb(pose_feat).unsqueeze(0).expand(B, -1, -1).reshape(B * self.nv, self.C)
        if self.no_pose_emb:
            cond = torch.zeros_like(cond)
        for intra, inter in zip(self.intra, self.inter):                  # alternating attention
            B_, N, M, C = F4.shape
            F4 = intra(F4.reshape(B_ * N, M, C), cond).reshape(B_, N, M, C)
            F4 = inter(F4, obs_valid)
        if self.no_cross:                                                 # ablation (a): mean over observations
            if not _mask_engaged(obs_valid):
                h = F4.mean(1)                                            # (B, M, C)
            else:
                w = obs_valid[:, :, None, None].to(F4.dtype)
                h = (F4 * w).sum(1) / w.sum(1).clamp_min(1)
        else:
            q = (self.q if self.no_ray_emb else self.q + self.dir_mlp(dir6).unsqueeze(0)).expand(B, -1, -1)
            h = self.ray_mic(q, F4.reshape(B, self.nv * self.M, self.C), dir3, poses, self.M, obs_valid)
        if not self.no_ray_emb:
            h = h + self.glob_dir(dir6).unsqueeze(0)
        for blk in self.erp:
            h = blk(h)
        return self._decode(h, fine, poses, obs_valid)
