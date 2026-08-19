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
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LH, LW = 16, 32                      # panoramic query grid (elevation x azimuth)

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

    def forward(self, Fm):                     # (B, N, M, C)
        B, N, M, C = Fm.shape
        x = Fm.permute(0, 2, 1, 3).reshape(B * M, N, C)
        o, _ = self.a(*[self.n(x)] * 3, need_weights=False)
        Fm = Fm + o.reshape(B, M, N, C).permute(0, 2, 1, 3)
        return Fm + self.m(Fm)


class RayMicAttn(nn.Module):
    """Geometry-aware cross-attention: panoramic ray queries attend all observation tokens. A per-(ray,
    observation) bias from [R_i^T r_j (3), ray . ear_axis (1), ear_sign (1)] is added to the logits before
    softmax. LN + residual + FFN."""
    def __init__(self, C, heads=8, use_bias=True):
        super().__init__()
        self.h, self.dk = heads, C // heads
        self.use_bias = use_bias          # False = "w/o geometric bias" ablation (content-only QK)
        self.nq = nn.LayerNorm(C); self.nk = nn.LayerNorm(C)
        self.q = nn.Linear(C, C); self.k = nn.Linear(C, C); self.v = nn.Linear(C, C); self.o = nn.Linear(C, C)
        self.bias_mlp = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, heads))
        self.ffn = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))

    def forward(self, q_in, tokens, ray_dir3, poses, M):
        B, R, C = q_in.shape; N = len(poses)
        Q = self.q(self.nq(q_in)).view(B, R, self.h, self.dk).transpose(1, 2)
        tk = self.nk(tokens)
        K = self.k(tk).view(B, -1, self.h, self.dk).transpose(1, 2)
        V = self.v(tk).view(B, -1, self.h, self.dk).transpose(1, 2)
        logits = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)
        if self.use_bias:
            bias = []
            for yaw, ear in poses:
                local = _yaw_rot_inv(ray_dir3, yaw)
                a = torch.tensor([math.cos(yaw), 0.0, -math.sin(yaw)], device=ray_dir3.device)  # ear axis R(yaw)@x_hat
                c = (ray_dir3 @ a).unsqueeze(-1) * ear
                e = torch.full_like(c, float(ear))
                bias.append(self.bias_mlp(torch.cat([local, c, e], -1)))
            bmic = torch.stack(bias, 1)                                    # (R,N,h)
            bfull = bmic.unsqueeze(2).expand(R, N, M, self.h).reshape(R, N * M, self.h)
            logits = logits + bfull.permute(2, 0, 1).unsqueeze(0)
        out = (logits.softmax(-1) @ V).transpose(1, 2).reshape(B, R, C)
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
                 no_pose_emb=False, no_ray_emb=False, no_geo_bias=False, no_tf_pe=False, no_cross=False):
        super().__init__()
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
        else:
            self.view_pose = _ALL_POSES[:nviews]                        # nviews=2 -> [0L,0R]; 4 -> [0L,0R,90R,270L]
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
        self.ray_mic = RayMicAttn(C, use_bias=not no_geo_bias)

    # ------------------------------------------------------------------ helpers
    def _pose_tensors(self, dev, view_pose=None):
        """-> pose_feat (nv,3) = [sin yaw, cos yaw, ear], poses list[(yaw, ear)]."""
        vp = view_pose if view_pose is not None else self.view_pose
        pose_feat = torch.tensor([[math.sin(y), math.cos(y), e] for y, e in vp], device=dev)
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

    def _fine_lift(self, fine_tok, poses, dev):
        """Per-observation ray-lift of the fine (2lh,2lw) tokens -> yaw roll-align -> mean. -> (B,C,2lh,2lw)."""
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
        return torch.stack(outs, 1).mean(1)

    def _decode(self, x_tok, fine_tok, poses):
        B = x_tok.size(0); dev = x_tok.device
        x = x_tok.transpose(1, 2).reshape(B, self.C, self.lh, self.lw)
        fine_erp = self._fine_lift(fine_tok, poses, dev)                  # (B,C,2lh,2lw)
        x = self.up_stages[0](x) + self.fine_to_dec(fine_erp)
        for st in self.up_stages[1:]:
            x = st(x)
        return torch.sigmoid(self.head(x))                                # depth / max_depth in [0,1]

    # ------------------------------------------------------------------ forward
    def forward(self, spec, view_poses=None):
        """spec (B, nviews, H, W) magnitude spectrograms; view_poses optional list[(yaw, ear)] of length
        nviews overriding the fixed mode poses. Returns depth in [0,1] (multiply by max_depth for metres)."""
        B = spec.size(0); dev = spec.device; dir6 = self.dir6_buf.to(dev); dir3 = dir6[:, :3]
        F4, poses, pose_feat, fine = self._encode(spec, view_poses)       # (B,nv,M,C)
        cond = self.pose_emb(pose_feat).unsqueeze(0).expand(B, -1, -1).reshape(B * self.nv, self.C)
        if self.no_pose_emb:
            cond = torch.zeros_like(cond)
        for intra, inter in zip(self.intra, self.inter):                  # alternating attention
            B_, N, M, C = F4.shape
            F4 = intra(F4.reshape(B_ * N, M, C), cond).reshape(B_, N, M, C)
            F4 = inter(F4)
        if self.no_cross:                                                 # ablation (a): mean over observations
            h = F4.mean(1)                                                # (B, M, C)
        else:
            q = (self.q if self.no_ray_emb else self.q + self.dir_mlp(dir6).unsqueeze(0)).expand(B, -1, -1)
            h = self.ray_mic(q, F4.reshape(B, self.nv * self.M, self.C), dir3, poses, self.M)
        if not self.no_ray_emb:
            h = h + self.glob_dir(dir6).unsqueeze(0)
        for blk in self.erp:
            h = blk(h)
        return self._decode(h, fine, poses)
