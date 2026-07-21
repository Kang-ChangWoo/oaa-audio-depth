"""OAA (Orientation-Aligned Alternating attention) — baseline depth model.

Binaural(+orientation) audio -> ERP depth (256x512, [0,1]*max_depth).

BASELINE = OAAv2Depth(cond_mode="adaln", nviews∈{2,4,6,8}) + masked-L1 + EMA.
This file is the CLEANED reference copy: all experimental features (IPD/GCC/ToF branches,
bins/uncertainty heads, pose heads, multi-scale lift, Fourier PE, subset machinery) are
removed. State-dict keys are IDENTICAL to the research repo's default-flag checkpoints,
so trained baselines (e.g. OAA_r8_adaln_s1) load with strict=True.

Input convention (loader order == pose order):
  nviews=2 : [0L, 0R]
  nviews=4 : cB = [0L, 0R, 90R, 270L]
  nviews=6 : [0L, 0R, 90L, 90R, 270L, 270R]
  nviews=8 : _POOL8 = [0L, 0R, 90L, 90R, 180L, 180R, 270L, 270R]
Arbitrary posed subsets: pass view_poses=[(yaw, ear), ...] to forward() and slice the
r8 input accordingly (pose-conditioning generalises zero-shot to unseen subsets).

Test MAE (Matterport3D, this data pipeline): 4ch 0.783 / 6ch 0.738 / 8ch 0.718 (SOTA).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LH, LW = 16, 32                       # coarse ERP token resolution (elevation x azimuth)

#              0L            0R            +90 (R ear)     -90 (L ear)
_ALL_POSES = [(0.0, -1.0), (0.0, 1.0), (math.pi / 2, 1.0), (-math.pi / 2, -1.0)]
# 8-channel orientation-ear pool (loader mode 'r8' order): (yaw, ear_sign), ear -1=L +1=R.
_POOL8 = [(0.0, -1.0), (0.0, 1.0), (math.pi / 2, -1.0), (math.pi / 2, 1.0),
          (math.pi, -1.0), (math.pi, 1.0), (-math.pi / 2, -1.0), (-math.pi / 2, 1.0)]
_CB_SUBSET = [0, 1, 3, 6]             # cB = r8[[0,1,3,6]]


def _dir_pe(h, w, device):
    """Unit direction per ERP cell -> [dx,dy,dz,dx^2,dy^2,dz^2]. d(A=0)=+z (front), d(+pi/2)=+x."""
    el = (torch.arange(h, device=device) + 0.5) / h * math.pi - math.pi / 2
    az = (torch.arange(w, device=device) + 0.5) / w * 2 * math.pi - math.pi
    el, az = torch.meshgrid(el, az, indexing="ij")
    d = torch.stack([torch.cos(el) * torch.sin(az), torch.sin(el), torch.cos(el) * torch.cos(az)], -1)
    d = torch.cat([d, d ** 2], -1)
    return d.reshape(h * w, 6)


def _yaw_rot_inv(dir3, yaw):
    """R(-yaw) applied to ray dirs (...,3): global ray -> mic-local frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy, dz = dir3[..., 0], dir3[..., 1], dir3[..., 2]
    return torch.stack([dx * c - dz * s, dy, dx * s + dz * c], -1)


def _norm(co, kind="group"):
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
    """Per-view CNN encoder (weight-shared across views). Learned strided stem sees full
    resolution once (time axis = echo delay = distance), then residual downsampling to (LH,LW)."""
    def __init__(self, C=256, ngf=64, in_ch=1, norm="group", lh=LH, lw=LW, enc_res=(128, 256)):
        super().__init__()
        self.enc_res = enc_res
        self.stem = nn.Sequential(nn.Conv2d(in_ch, ngf // 2, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(ngf // 2, ngf, 3, 1, 1))
        stages = int(round(math.log2(enc_res[0] / lh)))
        assert enc_res[0] // lh == enc_res[1] // lw == 2 ** stages, f"enc_res {enc_res} -> {lh}x{lw}"
        chans = [ngf, ngf * 2, ngf * 4, C, C, C][:stages] + [C]
        blocks = []
        for i in range(stages):
            blocks += [_ResBlk(chans[i], chans[i], down=False, norm=norm), _ResBlk(chans[i], chans[i + 1], norm=norm)]
        self.net = nn.Sequential(*blocks)
        self.C = C; self.lh, self.lw = lh, lw

    def forward(self, x):
        if x.shape[-2:] != tuple(2 * r for r in self.enc_res):          # stem expects 2*enc_res input
            x = F.interpolate(x, size=tuple(2 * r for r in self.enc_res), mode="bilinear", align_corners=False)
        h = self.stem(x)
        for blk in self.net:
            h = blk(h)
        return h.flatten(2).transpose(1, 2)                              # (B, lh*lw, C)


class SelfAttn(nn.Module):
    def __init__(self, C, heads=8):
        super().__init__()
        self.n = nn.LayerNorm(C); self.a = nn.MultiheadAttention(C, heads, batch_first=True)
        self.m = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))

    def forward(self, x):
        x = x + self.a(*[self.n(x)] * 3, need_weights=False)[0]
        return x + self.m(x)


class CondSelfAttn(nn.Module):
    """AdaLN-conditioned self-attention. Zero-init modulation -> initial behaviour == plain.
    THE architecture win of this project (pose-conditioned LayerNorm modulation, −0.02 MAE)."""
    def __init__(self, C, heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(C, elementwise_affine=False)
        self.a = nn.MultiheadAttention(C, heads, batch_first=True)
        self.n2 = nn.LayerNorm(C, elementwise_affine=False)
        self.m = nn.Sequential(nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))
        self.ada = nn.Linear(C, 6 * C)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, cond):                      # x: (B*N, M, C)  cond: (B*N, C) per-view pose emb
        sa, ba, ga, sm, bm, gm = self.ada(cond).unsqueeze(1).chunk(6, -1)
        h = self.n1(x) * (1 + sa) + ba
        x = x + ga * self.a(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + sm) + bm
        return x + gm * self.m(h)


class InterMicAttn(nn.Module):
    """Attention ACROSS views at each token position (views exchange evidence per ERP cell)."""
    def __init__(self, C, heads=8):
        super().__init__()
        self.n = nn.LayerNorm(C); self.a = nn.MultiheadAttention(C, heads, batch_first=True)
        self.m = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))

    def forward(self, Fm):                            # (B, N, M, C)
        B, N, M, C = Fm.shape
        x = Fm.permute(0, 2, 1, 3).reshape(B * M, N, C)
        o, _ = self.a(*[self.n(x)] * 3, need_weights=False)
        Fm = Fm + o.reshape(B, M, N, C).permute(0, 2, 1, 3)
        return Fm + self.m(Fm)


class RayMicAttn(nn.Module):
    """ERP ray queries cross-attend all view tokens with a per-(ray, mic) geometry bias from
    [ray in mic-local frame (3), ray·ear_axis (1), ear_sign (1)]."""
    def __init__(self, C, heads=8):
        super().__init__()
        self.h, self.dk = heads, C // heads
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
        bias = []
        for yaw, ear in poses:
            local = _yaw_rot_inv(ray_dir3, yaw)
            a = torch.tensor([math.cos(yaw), 0.0, -math.sin(yaw)], device=ray_dir3.device)  # ear axis R(yaw)@x
            c = (ray_dir3 @ a).unsqueeze(-1) * ear
            e = torch.full_like(c, float(ear))
            bias.append(self.bias_mlp(torch.cat([local, c, e], -1)))
        bmic = torch.stack(bias, 1)                                          # (R, N, h)
        bfull = bmic.unsqueeze(2).expand(R, N, M, self.h).reshape(R, N * M, self.h)
        logits = logits + bfull.permute(2, 0, 1).unsqueeze(0)
        out = (logits.softmax(-1) @ V).transpose(1, 2).reshape(B, R, C)
        h = q_in + self.o(out)
        return h + self.ffn(h)


def _make_up(C, norm, lh, lw):
    """Upsampling decoder (lh,lw) -> (256,512)."""
    stages = int(round(math.log2(256 / lh)))
    assert 256 % lh == 0 and 512 % lw == 0 and (256 // lh) == (512 // lw)
    chans = [C, 128, 64, 32, 16, 16, 16][:stages + 1]
    def up(ci, co):
        return nn.Sequential(nn.ConvTranspose2d(ci, co, 4, 2, 1), _norm(co, norm), nn.GELU(),
                             _ResBlk(co, co, down=False, norm=norm))
    return nn.Sequential(*[up(chans[i], chans[i + 1]) for i in range(stages)]), chans[stages]


class OAAv2Depth(nn.Module):
    """Baseline: per-view weight-shared encoder -> AdaLN pose conditioning -> alternating
    intra/inter-mic attention -> ray-mic geometry cross-attention -> ERP self-attention -> decoder.

    Params are nviews-invariant (~11.05M with adaln at C=256) — the orientation gain is free."""
    def __init__(self, C=256, rounds=2, in_ch=1, norm="group", lh=LH, lw=LW,
                 enc_res=(128, 256), nviews=4, cond_mode="adaln", max_depth=10.0):
        super().__init__()
        assert cond_mode in ("add", "adaln")
        self.C = C; self.in_ch = in_ch; self.nv = nviews
        self.lh, self.lw, self.M = lh, lw, lh * lw
        self.cond_mode = cond_mode; self.max_depth_n = max_depth
        self.enc = ViewEncoder(C, in_ch=in_ch, norm=norm, lh=lh, lw=lw, enc_res=enc_res)
        self.pose_emb = nn.Sequential(nn.Linear(3, C), nn.GELU(), nn.Linear(C, C))
        self.tf_pe = nn.Parameter(torch.zeros(1, self.M, C))
        self.q = nn.Parameter(torch.randn(1, self.M, C) * 0.02)
        self.dir_mlp = nn.Sequential(nn.Linear(6, C), nn.GELU(), nn.Linear(C, C))
        self.glob_dir = nn.Sequential(nn.Linear(6, C), nn.GELU(), nn.Linear(C, C))
        self.erp = nn.ModuleList([SelfAttn(C) for _ in range(4)])
        self.aux_head = nn.Linear(C, 1)               # kept for checkpoint compat (unused at inference)
        if nviews == 8:
            self.view_pose = list(_POOL8)
        elif nviews == 6:
            self.view_pose = [_POOL8[j] for j in (0, 1, 2, 3, 6, 7)]
        else:
            self.view_pose = _ALL_POSES[:nviews]
        self.register_buffer("dir6_buf", _dir_pe(lh, lw, torch.device("cpu")), persistent=False)
        self.intra = nn.ModuleList([(CondSelfAttn(C) if cond_mode == "adaln" else SelfAttn(C)) for _ in range(rounds)])
        self.inter = nn.ModuleList([InterMicAttn(C) for _ in range(rounds)])
        self.ray_mic = RayMicAttn(C)
        self.up, dec_out = _make_up(C, norm, lh, lw)
        self.head = nn.Conv2d(dec_out, 1, 3, padding=1)

    def _pose_feats(self, view_poses, dev):
        vp = view_poses if view_poses is not None else self.view_pose
        assert len(vp) == self.nv, f"view_poses len must be {self.nv}"
        pf = torch.tensor([[math.sin(y), math.cos(y), e] for y, e in vp], device=dev)
        return pf, list(vp)

    def forward(self, spec, view_poses=None):
        """spec (B, nviews*in_ch, 256, 512) magnitude STFT -> depth (B,1,256,512) in [0,1] (×max_depth = m)."""
        assert spec.size(1) == self.nv * self.in_ch, f"expected {self.nv * self.in_ch}ch, got {spec.size(1)}"
        B = spec.size(0); dev = spec.device
        dir6 = self.dir6_buf.to(dev); dir3 = dir6[:, :3]
        pf, poses = self._pose_feats(view_poses, dev)
        H, W = spec.shape[-2:]
        v = spec.view(B, self.nv, self.in_ch, H, W).reshape(B * self.nv, self.in_ch, H, W)
        t = self.enc(v).reshape(B, self.nv, self.M, self.C) + self.tf_pe.unsqueeze(1)
        t = t + self.pose_emb(pf).view(1, self.nv, 1, self.C)            # additive pose injection
        cond_bn = None
        if self.cond_mode == "adaln":                                    # AdaLN pose conditioning
            cond_bn = self.pose_emb(pf).unsqueeze(0).expand(B, -1, -1).reshape(B * self.nv, self.C)
        F4 = t
        for intra, inter in zip(self.intra, self.inter):
            B_, N, M, C = F4.shape
            xin = F4.reshape(B_ * N, M, C)
            F4 = (intra(xin, cond_bn) if self.cond_mode == "adaln" else intra(xin)).reshape(B_, N, M, C)
            F4 = inter(F4)
        tokens = F4.reshape(B, self.nv * self.M, self.C)
        q = (self.q + self.dir_mlp(dir6).unsqueeze(0)).expand(B, -1, -1)
        h = self.ray_mic(q, tokens, dir3, poses, self.M)
        h = h + self.glob_dir(dir6).unsqueeze(0)
        for blk in self.erp:
            h = blk(h)
        x2d = h.transpose(1, 2).reshape(B, self.C, self.lh, self.lw)
        return torch.sigmoid(self.head(self.up(x2d)))
