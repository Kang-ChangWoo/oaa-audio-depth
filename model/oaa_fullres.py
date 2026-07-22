"""Orientation-Aligned Alternating Attention (OAA) for cB 4-channel audio -> ERP depth.

Input channel contract (spec.size(1) == 4 * in_ch), order = per-view blocks:
  [v0 (0degL), v1 (0degR), v2 (+90deg-ear), v3 (-90deg-ear)]  each with in_ch feats
  in_ch=1: magnitude ; in_ch=3: [mag, cos(phi), sin(phi)] (phase-preserving).
Each of the 4 views is encoded INDEPENDENTLY by a weight-shared encoder (batched over the batch
dim, NOT stacked as input channels).

Coordinate convention (tests/test_oaa_geometry.py):
  _dir_pe azimuth A in [-pi, pi); d(A) = (sinA, 0, cosA). A=0 -> +z (front), A=+pi/2 -> +x.
  yaw-psi rig forward = d(psi); global ray local azimuth = A_global - psi.
  local->global alignment roll = +roll_sign*psi*W/(2pi) cols (v1); global->local = R(-psi)=R^T (v2).

pose/roll are computed DYNAMICALLY from view_pose (+ optional azimuth-rotation aug az_k), not
buffered. shuffle_pose control: per-batch perm; training uses the GLOBAL RNG (a fresh perm every
forward); eval is reproducible via shuffle_eval_seed. Azimuth aug (az_k) is applied in the trainer.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LH, LW = 16, 32                      # lift / coarse-ERP resolution (elevation x azimuth); GPU-budget tuned
assert LW % 4 == 0, "LW must be divisible by 4 so +-90deg roll is an integer column count"

#                0L            0R            +90 (R ear)      -90 (L ear)    -- nviews=2 -> plain binaural [0L,0R]
_ALL_POSES = [(0.0, -1.0), (0.0, 1.0), (math.pi / 2, 1.0), (-math.pi / 2, -1.0)]
# 8-channel orientation pool (loader mode 'r8' order): (yaw, ear_sign), ear -1=L +1=R.
# subset training picks any 4 of these with their TRUE poses; predict 0deg-world-frame depth.
_POOL8 = [(0.0, -1.0), (0.0, 1.0), (math.pi / 2, -1.0), (math.pi / 2, 1.0),
          (math.pi, -1.0), (math.pi, 1.0), (-math.pi / 2, -1.0), (-math.pi / 2, 1.0)]
_CB_SUBSET = [0, 1, 3, 6]             # cB-equivalent 4 = [0L, 0R, 90R, 270L] (for reproducible eval)
_MS_POSES = [(0.0, 0.0), (0.0, 1.0)]  # D: magnitude Mid/Side pair -> sum(ear=0, no L/R), diff(ear=+1, ILD axis)
_MIRROR8 = [1, 0, 7, 6, 5, 4, 3, 2]   # v7: LR-mirror (yaw,e)->(-yaw,-e) as a pool8 index permutation
_VIEW_OFFSETS = {2: [0, 0], 4: [0, 0, 1, 3], 6: [0, 0, 1, 1, 3, 3], 8: [0, 0, 1, 1, 2, 2, 3, 3]}  # view i -> yaw offset (0=0deg,1=90,2=180,3=270)


def assemble_ipd(spec, ipd_ang, nviews, sym=False):
    """spec (B,N,H,W) magnitude + ipd_ang (B,4,Fn,Tn) native per-yaw-offset IPD angle ->
    (B, N*3, H, W): per-view groups [mag, cos IPD, sin IPD] for in_ch=3 encoding. Each view gets
    the IPD of ITS OWN yaw (both ears of a yaw share it). Nearest resize = the magnitude pipeline's.
    sym=True (LR-agnostic): cos only -> in_ch=2. cos is EVEN under ear swap (keeps |phase-diff| =
    how far off-axis); sin is ODD (it IS the side label), so it must be dropped when L/R is unknown."""
    B, N, H, W = spec.shape
    ch = [torch.cos(ipd_ang.float())] if sym else [torch.cos(ipd_ang.float()), torch.sin(ipd_ang.float())]
    nc = len(ch)
    cs = torch.stack(ch, 2)                                                            # (B,4,nc,Fn,Tn)
    cs = F.interpolate(cs.flatten(1, 2), size=(H, W), mode="nearest").view(B, 4, nc, H, W).to(spec.dtype)
    off = _VIEW_OFFSETS[N]
    return torch.cat([torch.cat([spec[:, i:i + 1], cs[:, off[i]]], 1) for i in range(N)], 1)


def _dir_pe(h, w, device):
    """Unit direction per ERP cell -> [dx,dy,dz,dx^2,dy^2,dz^2]. d(A=0)=+z (front), d(+pi/2)=+x."""
    el = (torch.arange(h, device=device) + 0.5) / h * math.pi - math.pi / 2
    az = (torch.arange(w, device=device) + 0.5) / w * 2 * math.pi - math.pi
    E, A = torch.meshgrid(el, az, indexing="ij")
    dx = torch.cos(E) * torch.sin(A); dy = torch.sin(E); dz = torch.cos(E) * torch.cos(A)
    d = torch.stack([dx, dy, dz, dx * dx, dy * dy, dz * dz], -1)
    return d.reshape(h * w, 6)


def _yaw_rot_inv(dir3, yaw):
    """R(-yaw) applied to ray dirs (...,3): global ray -> mic-local frame.  d(A) -> d(A - yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy, dz = dir3[..., 0], dir3[..., 1], dir3[..., 2]
    return torch.stack([dx * c - dz * s, dy, dx * s + dz * c], -1)


def _fourier_dir(d, K=4):
    """(...,3) unit dirs -> (..., 3 + 6K). raw kept + multi-frequency sin/cos (B1: sharper azimuth).
    dir3/dir6 have adj-cell cos-sim ~0.98 (indistinguishable at 11.25deg); Fourier K=4 -> ~0.59."""
    fr = (2.0 ** torch.arange(K, device=d.device, dtype=d.dtype)) * math.pi
    ang = d.unsqueeze(-1) * fr                       # (...,3,K)
    return torch.cat([d, ang.sin().flatten(-2), ang.cos().flatten(-2)], -1)


def _pe_indim(pe_mode, pe_K, base):
    """input dim for a PE-consuming MLP. raw: `base` (dir6=6 or [local3+c+e]=5 etc). fourier: 3+6K (+extras)."""
    return base if pe_mode == "raw" else (3 + 6 * pe_K) + (base - 3)


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
    """RayDPT-level encoder. LEARNED strided stem sees full-res once (time axis = echo-delay =
    distance -> no free-lunch bilinear downsample), then residual downsampling to (LH, LW)."""
    def __init__(self, C=256, ngf=64, in_ch=1, norm="group", lh=LH, lw=LW, enc_res=(128, 256)):
        super().__init__()
        self.enc_res = enc_res
        self.stem = nn.Sequential(nn.Conv2d(in_ch, ngf // 2, 3, 2, 1), nn.GELU(),   # full-res -> 1/2 (learned)
                                  nn.Conv2d(ngf // 2, ngf, 3, 1, 1))
        stages = int(round(math.log2(enc_res[0] / lh)))                             # post-stem downsamplings
        assert enc_res[0] // lh == enc_res[1] // lw == 2 ** stages, f"enc_res {enc_res} -> {lh}x{lw}"
        chans = [ngf, ngf * 2, ngf * 4, C, C, C][:stages] + [C]
        blocks = []
        for i in range(stages):
            blocks += [_ResBlk(chans[i], chans[i], down=False, norm=norm), _ResBlk(chans[i], chans[i + 1], norm=norm)]
        self.net = nn.Sequential(*blocks); self.C = C
        self.lh, self.lw = lh, lw
        self.fine_ch = chans[stages - 1] if stages >= 2 else C     # channel at (2lh,2lw) tap (multi-scale)

    def forward(self, x, fine=False):
        if x.shape[-2:] != tuple(2 * r for r in self.enc_res):                      # stem expects 2*enc_res input
            x = F.interpolate(x, size=tuple(2 * r for r in self.enc_res), mode="bilinear", align_corners=False)
        h = self.stem(x); ft = None
        for blk in self.net:
            h = blk(h)
            if fine and h.shape[-2:] == (2 * self.lh, 2 * self.lw):                  # tap fine scale (2lh,2lw)
                ft = h.flatten(2).transpose(1, 2)                                    # (B, 2lh*2lw, C)
        tok = h.flatten(2).transpose(1, 2)                                           # (B, lh*lw, C)
        return (tok, ft) if fine else tok


class SelfAttn(nn.Module):
    def __init__(self, C, heads=8):
        super().__init__()
        self.n = nn.LayerNorm(C); self.a = nn.MultiheadAttention(C, heads, batch_first=True)
        self.m = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))

    def forward(self, x):
        x = x + self.a(*[self.n(x)] * 3, need_weights=False)[0]; return x + self.m(x)


class GatedViewFuse(nn.Module):
    """Per-ray, a fuse query attends the N views; each view gets a logit bias from the ray direction
    seen in THAT view's local frame (R_i^T r_j). Residual mean path for fast early training."""
    def __init__(self, C, heads=8, pe_mode="raw", pe_K=4):
        super().__init__()
        self.h, self.dk = heads, C // heads; self.pe_mode = pe_mode; self.pe_K = pe_K
        self.q = nn.Linear(C, C); self.k = nn.Linear(C, C); self.v = nn.Linear(C, C); self.o = nn.Linear(C, C)
        self.fuse = nn.Parameter(torch.randn(1, 1, C) * 0.02)
        self.gate = nn.Sequential(nn.Linear(_pe_indim(pe_mode, pe_K, 3), 64), nn.GELU(), nn.Linear(64, heads))
        self.norm = nn.LayerNorm(C)

    def forward(self, Sstack, local_dirs):
        B, N, R, C = Sstack.shape
        x = self.norm(Sstack).permute(0, 2, 1, 3).reshape(B * R, N, C)
        Q = self.q(self.fuse).view(1, 1, self.h, self.dk).transpose(1, 2).expand(B * R, -1, -1, -1)
        K = self.k(x).view(B * R, N, self.h, self.dk).transpose(1, 2)
        V = self.v(x).view(B * R, N, self.h, self.dk).transpose(1, 2)
        logits = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)
        gate_in = local_dirs if self.pe_mode == "raw" else _fourier_dir(local_dirs, self.pe_K)
        bias = self.gate(gate_in).permute(1, 2, 0).reshape(1, R, self.h, 1, N)
        bias = bias.expand(B, -1, -1, -1, -1).reshape(B * R, self.h, 1, N)
        out = (logits + bias).softmax(-1) @ V
        out = self.o(out.transpose(1, 2).reshape(B * R, 1, C)).reshape(B, R, C)
        return out + Sstack.mean(1)                                  # residual (Task 6)


class InterMicAttn(nn.Module):
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
    """ERP ray queries cross-attend multichannel tokens with per-(ray,mic) bias from
    [R_i^T r_j (3), ray.ear_axis (1), ear_sign (1)]. LN + residual + FFN."""
    def __init__(self, C, heads=8, pe_mode="raw", pe_K=4):
        super().__init__()
        self.h, self.dk = heads, C // heads; self.pe_mode = pe_mode; self.pe_K = pe_K
        self.nq = nn.LayerNorm(C); self.nk = nn.LayerNorm(C)
        self.q = nn.Linear(C, C); self.k = nn.Linear(C, C); self.v = nn.Linear(C, C); self.o = nn.Linear(C, C)
        self.bias_mlp = nn.Sequential(nn.Linear(_pe_indim(pe_mode, pe_K, 5), 64), nn.GELU(), nn.Linear(64, heads))
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
            a = torch.tensor([math.cos(yaw), 0.0, -math.sin(yaw)], device=ray_dir3.device)  # ear axis R(yaw)@x_hat
            c = (ray_dir3 @ a).unsqueeze(-1) * ear
            e = torch.full_like(c, float(ear))
            lf = local if self.pe_mode == "raw" else _fourier_dir(local, self.pe_K)
            bias.append(self.bias_mlp(torch.cat([lf, c, e], -1)))
        bmic = torch.stack(bias, 1)                                        # (R,N,h)
        bfull = bmic.unsqueeze(2).expand(R, N, M, self.h).reshape(R, N * M, self.h)
        logits = logits + bfull.permute(2, 0, 1).unsqueeze(0)
        out = (logits.softmax(-1) @ V).transpose(1, 2).reshape(B, R, C)
        h = q_in + self.o(out)
        return h + self.ffn(h)


class CondSelfAttn(nn.Module):
    """B2: AdaLN-conditioned self-attn. zero-init gate -> initial behaviour == plain (identity of cond)."""
    def __init__(self, C, heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(C, elementwise_affine=False)
        self.a = nn.MultiheadAttention(C, heads, batch_first=True)
        self.n2 = nn.LayerNorm(C, elementwise_affine=False)
        self.m = nn.Sequential(nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))
        self.ada = nn.Linear(C, 6 * C)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, cond):                      # x: (B*, M, C)  cond: (B*, C) per-view pose_emb
        sa, ba, ga, sm, bm, gm = self.ada(cond).unsqueeze(1).chunk(6, -1)
        h = self.n1(x) * (1 + sa) + ba
        x = x + ga * self.a(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + sm) + bm
        return x + gm * self.m(h)


class ERPConv2d(nn.Module):
    """ERP-correct conv: azimuth(수평)는 circular, elevation(수직)은 replicate padding."""
    def __init__(self, ci, co, k=3, bias=True):
        super().__init__()
        assert k % 2 == 1
        self.p = k // 2
        self.conv = nn.Conv2d(ci, co, k, 1, 0, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.p, self.p, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.p, self.p), mode="replicate")
        return self.conv(x)


class ERPResBlk(nn.Module):
    def __init__(self, ci, co, norm="group"):
        super().__init__()
        self.c1 = ERPConv2d(ci, co, 3); self.n1 = _norm(co, norm)
        self.c2 = ERPConv2d(co, co, 3); self.n2 = _norm(co, norm)
        self.sc = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x))); h = self.n2(self.c2(h))
        return F.gelu(h + self.sc(x))


def _make_up_resize_erp(C, norm, lh, lw):
    """artifact-reduced 디코더: bilinear resize + ERP conv (checkerboard 제거 + seam 위생).
    실측 근거: ConvT 디코더의 수평 고주파 교대율 0.266 vs GT 0.073 (2026-07-22)."""
    stages = int(round(math.log2(256 / lh)))
    assert 256 % lh == 0 and 512 % lw == 0 and (256 // lh) == (512 // lw)
    chans = [C, 128, 64, 32, 16, 16, 16][:stages + 1]
    def up(ci, co):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                             ERPConv2d(ci, co, 3), _norm(co, norm), nn.GELU(), ERPResBlk(co, co, norm=norm))
    return nn.Sequential(*[up(chans[i], chans[i + 1]) for i in range(stages)]), chans[stages]


def _make_up(C, norm, lh, lw, deep=False, as_stages=False):
    """upsample (lh,lw) -> (256,512). #stages = log2(256/lh). deep=+1 refine ResBlk per stage.
    as_stages -> return ModuleList of per-stage blocks (for mid-decoder multi-scale injection) + their out-channels."""
    import math as _m
    stages = int(round(_m.log2(256 / lh)))
    assert 256 % lh == 0 and 512 % lw == 0 and (256 // lh) == (512 // lw), f"bad lift res {lh}x{lw}"
    chans = [C, 128, 64, 32, 16, 16, 16][:stages + 1]
    def up(ci, co):
        blks = [nn.ConvTranspose2d(ci, co, 4, 2, 1), _norm(co, norm), nn.GELU(), _ResBlk(co, co, down=False, norm=norm)]
        if deep: blks.append(_ResBlk(co, co, down=False, norm=norm))
        return nn.Sequential(*blks)
    mods = [up(chans[i], chans[i + 1]) for i in range(stages)]
    if as_stages:
        return nn.ModuleList(mods), chans[1:stages + 1]              # per-stage out channels
    return nn.Sequential(*mods), chans[stages]


class _OAABase(nn.Module):
    def __init__(self, C, in_ch, norm, roll_sign, lh=LH, lw=LW, dec_deep=False, enc_res=(128, 256), nviews=4,
                 pe_mode="raw", pe_K=4, multi_scale_lift=False,
                 head_mode="regress", n_bins=64, min_depth=0.1, max_depth=10.0, unc_mode="none", sigma_b=1.0,
                 tof_mode="none", tof_bins=16, pose_head=False, pose_cls=False, dec_mode="convt"):
        super().__init__()
        self.C = C; self.in_ch = in_ch; self.roll_sign = roll_sign; self.nv = nviews
        self.lh, self.lw, self.M = lh, lw, lh * lw; self.enc_res = enc_res
        self.pe_mode = pe_mode; self.pe_K = pe_K; self.multi_scale_lift = multi_scale_lift
        self.head_mode = head_mode; self.n_bins = n_bins; self.unc_mode = unc_mode; self.sigma_b = sigma_b
        self.max_depth_n = max_depth; self.tof_mode = tof_mode; self.tof_time_shuffle = False; self.echo_sr = 48000
        assert dec_mode in ("convt", "resize_erp"); self.dec_mode = dec_mode
        assert pe_mode in ("raw", "fourier"), pe_mode
        assert head_mode in ("regress", "bins") and unc_mode in ("none", "laplace") and tof_mode in ("none", "pool", "echo", "gcc")
        self.shuffle_eval_seed = 0
        self.enc = ViewEncoder(C, in_ch=in_ch, norm=norm, lh=lh, lw=lw, enc_res=enc_res)
        self.pose_emb = nn.Sequential(nn.Linear(3, C), nn.GELU(), nn.Linear(C, C))
        self.pose_head = pose_head                                       # VGGT-style: infer 2D yaw direction from audio
        if pose_head:
            self.pose_predictor = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 2))  # -> [sin_yaw,cos_yaw] (2D only)
            self._pose_pred = None; self._pose_target = None
        self.pose_cls = pose_cls                                         # 8-way channel-identity classification (pool8 slot)
        if pose_cls:
            self.pose_cls_head = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 8))
            self._pose_cls_logits = None
        self.tf_pe = nn.Parameter(torch.zeros(1, self.M, C))
        self.q = nn.Parameter(torch.randn(1, self.M, C) * 0.02)
        dir_in = 6 if pe_mode == "raw" else 3 + 6 * pe_K              # raw dir6 vs fourier(dir3) (sq terms subsumed)
        self.dir_mlp = nn.Sequential(nn.Linear(dir_in, C), nn.GELU(), nn.Linear(C, C))
        self.glob_dir = nn.Sequential(nn.Linear(dir_in, C), nn.GELU(), nn.Linear(C, C))
        self.erp = nn.ModuleList([SelfAttn(C) for _ in range(4)])
        self.aux_head = nn.Linear(C, 1)                                  # coarse aux supervision (Task 5)
        if nviews == 8:                                                  # loader mode 'r8' channel order
            self.view_pose = list(_POOL8)
        elif nviews == 6:                                                # loader mode 'r6' order: 0LR + 90LR + 270LR
            self.view_pose = [_POOL8[j] for j in (0, 1, 2, 3, 6, 7)]
        else:
            self.view_pose = _ALL_POSES[:nviews]                        # nviews=2 -> [0L,0R] plain binaural
        self.register_buffer("dir6_buf", _dir_pe(lh, lw, torch.device("cpu")), persistent=False)
        if multi_scale_lift:                                             # cause-1 fix: fine (2lh,2lw) ray-lift skip
            assert int(round(math.log2(enc_res[0] / lh))) >= 2, \
                "multi_scale_lift requires >=2 encoder stages (fine tap at (2lh,2lw))"   # P1-b
            self.up_stages, up_ch = _make_up(C, norm, lh, lw, deep=dec_deep, as_stages=True)
            dec_out = up_ch[-1]
            self.fine_in = nn.Linear(self.enc.fine_ch, C)               # project fine audio tokens -> C
            self.fine_q = nn.Parameter(torch.randn(1, 4 * self.M, C) * 0.02)
            self.fine_tf_pe = nn.Parameter(torch.zeros(1, 4 * self.M, C))   # P1-a: fine KV TF-PE (time=echo-delay=dist)
            self.fine_dir_mlp = nn.Sequential(nn.Linear(dir_in, C), nn.GELU(), nn.Linear(C, C))
            self.fine_ln = nn.LayerNorm(C)
            self.fine_lift = nn.MultiheadAttention(C, 8, batch_first=True)
            self.fine_to_dec = nn.Conv2d(C, up_ch[0], 1)               # C -> stage-0 out ch; zero-init = start no-inject
            nn.init.zeros_(self.fine_to_dec.weight); nn.init.zeros_(self.fine_to_dec.bias)
            self.register_buffer("fine_dir6_buf", _dir_pe(2 * lh, 2 * lw, torch.device("cpu")), persistent=False)
        else:
            if getattr(self, "dec_mode", "convt") == "resize_erp":
                self.up, dec_out = _make_up_resize_erp(C, norm, lh, lw)
            else:
                self.up, dec_out = _make_up(C, norm, lh, lw, deep=dec_deep)
        # A/B: depth head (regress=1ch sigmoid | bins=n_bins softmax soft-argmax) + optional laplace sigma
        if getattr(self, "dec_mode", "convt") == "resize_erp":
            self.head = ERPConv2d(dec_out, (n_bins if head_mode == "bins" else 1), 3)
        else:
            self.head = nn.Conv2d(dec_out, (n_bins if head_mode == "bins" else 1), 3, padding=1)
        if head_mode == "bins":
            d = torch.exp(torch.linspace(math.log(min_depth), math.log(max_depth), n_bins))
            self.register_buffer("bin_centers", d / max_depth, persistent=True)   # (K,) in [0,1]
        if unc_mode == "laplace":
            self.sigma_head = nn.Conv2d(dec_out, 1, 3, padding=1)
        if tof_mode == "pool":                                            # C: ToF evidence pooling (time=2d/c prior)
            dk = torch.exp(torch.linspace(math.log(min_depth), math.log(max_depth), tof_bins))
            self.register_buffer("tof_depths", dk, persistent=True)       # (K,) meters
            self.tof_t0 = nn.Parameter(torch.zeros(1))                    # T1: onset/window offset (init from T0's b)
            self.tof_demb = nn.Parameter(torch.zeros(tof_bins, C))        # T1: depth-bin identity (zero-init)
            self.tof_log_sigma = nn.Parameter(torch.tensor(0.4))         # T1: sigma=softplus+0.25 ~1.5tok; collapse guard
            self.tof_r = nn.Linear(C, C); self.tof_b = nn.Linear(C, C); self.tof_out = nn.Linear(C, C)
            nn.init.zeros_(self.tof_out.weight); nn.init.zeros_(self.tof_out.bias)   # zero-init -> start no-change
        if tof_mode == "echo":                                            # C (correct): waveform matched-filter ToF
            from reference_modules import Echo1DEncoder
            self.echo_enc = Echo1DEncoder(in_ch=4, channels=(64, 128, 256), out_dim=256)  # echo_channels -> 4ch
            self.echo_proj = nn.Linear(256, C)
            self.echo_gate = nn.Parameter(torch.zeros(1))                # zero-init inject -> start no-change
            self.register_buffer("echo_ref", torch.zeros(144), persistent=True)   # template, set by trainer/eval
        if tof_mode == "gcc":                                            # framewise GCC-PHAT ITD-vs-time map (0deg pair)
            self.gcc_enc = nn.Sequential(
                nn.Conv2d(1, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.gcc_proj = nn.Linear(128, C)
            self.gcc_gate = nn.Parameter(torch.zeros(1))                 # zero-init inject -> start no-change

    def _perm(self):
        if not self.shuffle_pose:
            return torch.arange(self.nv)
        if self.training:
            return torch.randperm(self.nv)                              # global RNG: fresh perm each forward
        g = torch.Generator(); g.manual_seed(self.shuffle_eval_seed)     # eval reproducible (seed configurable)
        return torch.randperm(self.nv, generator=g)

    def _pose_tensors(self, perm, az_k, dev, view_pose=None):
        """perm (nv,), az_k int (lift cols). -> pose_feat (nv,3), shifts (nv,) long, poses list[(yaw,ear)].
        view_pose overrides self.view_pose (random-subset training passes the sampled views' true poses)."""
        vp = view_pose if view_pose is not None else self.view_pose
        dpsi = az_k * 2 * math.pi / self.lw
        yaws, ears = [], []
        for i in perm.tolist():
            y, e = vp[i]; yaws.append(y + dpsi); ears.append(e)
        pose_feat = torch.tensor([[math.sin(y), math.cos(y), e] for y, e in zip(yaws, ears)], device=dev)
        shifts = torch.tensor([int(round(self.roll_sign * y / (2 * math.pi) * self.lw)) for y in yaws], device=dev)
        return pose_feat, shifts, list(zip(yaws, ears))

    def _encode(self, spec, az_k, view_pose=None):
        """batched per-view encode (NOT channel-stacked) + TF PE + pose emb. -> (B,nv,M,C), poses, shifts, fine.
        fine = (B,nv,4M,fine_ch) at (2lh,2lw) when multi_scale_lift else None.
        view_pose (len == nv) overrides the fixed config for random-subset training."""
        assert view_pose is None or len(view_pose) == self.nv, f"view_pose len must be {self.nv}"
        assert spec.size(1) == self.nv * self.in_ch, f"expected {self.nv * self.in_ch}ch, got {spec.size(1)}"
        B = spec.size(0); dev = spec.device; H, W = spec.shape[-2:]
        perm = self._perm().to(dev)
        pose_feat, shifts, poses = self._pose_tensors(perm, az_k, dev, view_pose)
        v = spec.view(B, self.nv, self.in_ch, H, W).reshape(B * self.nv, self.in_ch, H, W)
        fine = None
        if self.multi_scale_lift:
            enc_t, fine_t = self.enc(v, fine=True)
            t = enc_t.reshape(B, self.nv, self.M, self.C) + self.tf_pe.unsqueeze(1)
            fine = fine_t.reshape(B, self.nv, 4 * self.M, self.enc.fine_ch)
        else:
            t = self.enc(v).reshape(B, self.nv, self.M, self.C) + self.tf_pe.unsqueeze(1)
        if self.pose_head:                                              # infer 2D yaw from audio (pre-injection tokens)
            self._pose_pred = self.pose_predictor(t.mean(2))            # (B,nv,2) predicted [sin_yaw,cos_yaw]
            self._pose_target = pose_feat[:, :2]                       # (nv,2) known [sin_yaw,cos_yaw] (ear dropped)
        if self.pose_cls:                                               # classify pool8 identity (pre-injection tokens)
            self._pose_cls_logits = self.pose_cls_head(t.mean(2))       # (B,nv,8)
        t = t + self.pose_emb(pose_feat).view(1, self.nv, 1, self.C)   # pipeline still uses KNOWN pose (Phase 1)
        return t, poses, shifts, fine

    def _fine_lift(self, fine_tok, poses, dev):
        """per-view directional lift of fine (2lh,2lw) audio tokens -> roll-align -> mean. -> (B,C,2lh,2lw)."""
        B, nv, R2, _ = fine_tok.shape
        d6 = self.fine_dir6_buf.to(dev)                                   # (R2,6)
        q = (self.fine_q + self.fine_dir_mlp(self._dirfeat(d6)).unsqueeze(0)).expand(B * nv, -1, -1)
        kv = self.fine_in(fine_tok).reshape(B * nv, R2, self.C) + self.fine_tf_pe    # P1-a: fine KV TF-PE
        s = self.fine_lift(self.fine_ln(q), kv, kv, need_weights=False)[0].reshape(B, nv, R2, self.C)
        outs = []
        for i, (yaw, ear) in enumerate(poses):
            sh = int(round(self.roll_sign * yaw / (2 * math.pi) * (2 * self.lw)))    # 2x-res align shift
            si = s[:, i].transpose(1, 2).reshape(B, self.C, 2 * self.lh, 2 * self.lw)
            outs.append(torch.roll(si, shifts=sh, dims=-1))
        return torch.stack(outs, 1).mean(1)

    def _apply_head(self, x, want_sigma=False):
        """decoder feature (B,dec_out,256,512) -> (out depth [0,1], logits|None, sigma|None).
        bins: softmax over depth bins then soft-argmax. laplace: softplus sigma map (only when want_sigma)."""
        h = self.head(x); logits = None; sigma = None
        if self.head_mode == "bins":
            logits = h
            out = (h.softmax(1) * self.bin_centers.view(1, -1, 1, 1)).sum(1, keepdim=True)
        else:
            out = torch.sigmoid(h)
        if self.unc_mode == "laplace" and want_sigma:                     # P2: skip sigma conv at eval
            sigma = F.softplus(self.sigma_head(x)) + 1e-3
        return out, logits, sigma

    def _decode_ms(self, x_tok, fine_tok, poses, return_aux):
        """multi-scale decode: stage0 up (lh,lw)->(2lh,2lw), inject fine ray-lift skip, then remaining stages."""
        B = x_tok.size(0); dev = x_tok.device
        coarse = torch.sigmoid(self.aux_head(x_tok)).transpose(1, 2).reshape(B, 1, self.lh, self.lw) if return_aux else None
        x = x_tok.transpose(1, 2).reshape(B, self.C, self.lh, self.lw)
        fine_erp = self._fine_lift(fine_tok, poses, dev)                  # (B,C,2lh,2lw)
        x = self.up_stages[0](x) + self.fine_to_dec(fine_erp)            # inject (zero-init -> starts at 0)
        for st in self.up_stages[1:]:
            x = st(x)
        out, logits, sigma = self._apply_head(x, want_sigma=return_aux)
        return (out, coarse, logits, sigma) if return_aux else out

    def _tof(self, h, F4):
        """C: pool audio evidence at expected echo-token-time t*_k(depth) then ray attends bins. Pose-independent
        (round-trip 2d/c). Corrected mapping: token-col/lw = depth/max_depth (CUT = round-trip of max_depth)."""
        B, nv, M, C = F4.shape; dev = h.device
        E = F4.reshape(B, nv, self.lh, self.lw, C).mean(2)               # freq-avg -> (B,nv,lw,C); lw = time axis
        if self.tof_time_shuffle:                                        # C2/T2 control: break time->depth prior
            assert not self.training, "tof_time_shuffle is an eval-only control"
            g = torch.Generator(device="cpu"); g.manual_seed(0)          # reproducible fixed perm
            E = E[:, :, torch.randperm(self.lw, generator=g).to(dev)]
        t = torch.arange(self.lw, device=dev).float()
        tstar = (self.tof_depths / self.max_depth_n) * self.lw + self.tof_t0     # T1: +learnable onset offset
        sig = F.softplus(self.tof_log_sigma) + 0.25                     # T1: min-width guard (~1.5 token)
        w = (-(t.view(1, -1) - tstar.view(-1, 1)) ** 2 / (2 * sig ** 2)).softmax(-1)  # (K,lw)
        z = torch.einsum('kt,bntc->bnkc', w, E).mean(1) + self.tof_demb.unsqueeze(0)  # T1: pool + bin identity
        s = (self.tof_r(h) @ self.tof_b(z).transpose(-2, -1)) / math.sqrt(C)   # ray-bin evidence (B,R,K)
        return h + self.tof_out(s.softmax(-1) @ z)                      # zero-init inject -> starts unchanged

    def _echo(self, h, wave):
        """C (correct): waveform matched-filter ToF -> global radial feature injected into ERP tokens.
        ToF lives in the WAVEFORM (phase/timing), NOT magnitude spec (T0 confirmed) -> matched_filter."""
        from reference_modules import matched_filter, echo_channels
        ml = int(2 * self.max_depth_n / 343.0 * self.echo_sr)            # max round-trip lag (samples)
        echo = matched_filter(wave, self.echo_ref, norm="energy", max_lag=ml)
        ev = self.echo_proj(self.echo_enc(echo_channels(echo)))         # (B,C) global ToF/radial evidence
        return h + self.echo_gate * ev.unsqueeze(1)                     # zero-init gate

    def _gcc(self, h, wave):
        """framewise GCC-PHAT between the 0deg L/R waveforms -> ITD-vs-time map (echo direction vs
        round-trip distance). Direct sound pins lag~0; off-zero lag energy over time is the interaural
        TIMING cue that magnitude input discards (waveform sibling of the IPD input)."""
        fr = wave[:, :2].float().unfold(-1, 256, 128)                    # (B,2,T24,256) 5.3ms frames
        X = torch.fft.rfft(fr, n=512)
        Cx = X[:, 0] * X[:, 1].conj()
        Cx = Cx / (Cx.abs() + 1e-8)                                      # PHAT whitening
        cc = torch.fft.irfft(Cx, n=512)                                  # (B,T24,512), lag 0 at col 0
        cc = torch.cat([cc[..., -64:], cc[..., :65]], -1)                # lags [-64,64] (~±1.3ms interaural range)
        if getattr(self, "gcc_sym", False):                              # LR-agnostic: fold ±lag (swap-invariant)
            cc = cc[..., 64:] + cc[..., :65].flip(-1)                    # |lag| profile, side info removed
        ev = self.gcc_proj(self.gcc_enc(cc.unsqueeze(1).to(h.dtype)))    # (B,C)
        return h + self.gcc_gate * ev.unsqueeze(1)                       # zero-init gate

    def _dirfeat(self, dir6):
        """PE feature for dir_mlp/glob_dir: raw dir6 (6) or fourier(dir3) (3+6K). computed per forward."""
        return dir6 if self.pe_mode == "raw" else _fourier_dir(dir6[:, :3], self.pe_K)

    def _decode(self, x_tok, return_aux):
        B = x_tok.size(0)
        coarse = torch.sigmoid(self.aux_head(x_tok)).transpose(1, 2).reshape(B, 1, self.lh, self.lw) if return_aux else None
        x2d = x_tok.transpose(1, 2).reshape(B, self.C, self.lh, self.lw)
        out, logits, sigma = self._apply_head(self.up(x2d), want_sigma=return_aux)
        return (out, coarse, logits, sigma) if return_aux else out


class OAADepth(_OAABase):
    """v1: lift + deterministic roll alignment + gated per-(ray,view) fusion."""
    def __init__(self, C=256, variant="align_attn", shuffle_pose=False, in_ch=1, norm="group", roll_sign=1,
                 lh=LH, lw=LW, dec_deep=False, enc_res=(128, 256), nviews=4, pe_mode="raw", pe_K=4, **hkw):
        assert variant in ("noalign", "align_avg", "align_attn"), \
            f"variant '{variant}' not handled by OAADepth (concat is a separate baseline)"
        super().__init__(C, in_ch, norm, roll_sign, lh, lw, dec_deep, enc_res, nviews, pe_mode, pe_K, **hkw)
        self.variant = variant; self.shuffle_pose = shuffle_pose
        self.lift_norm = nn.LayerNorm(C); self.lift = nn.MultiheadAttention(C, 8, batch_first=True)
        self.view_blocks = nn.ModuleList([SelfAttn(C) for _ in range(3)])
        self.view_fuse = GatedViewFuse(C, pe_mode=pe_mode, pe_K=pe_K) if variant == "align_attn" else None

    def _lift(self, tokens, dir6):
        q = (self.q + self.dir_mlp(self._dirfeat(dir6)).unsqueeze(0)).expand(tokens.size(0), -1, -1)
        return self.lift(self.lift_norm(q), tokens, tokens, need_weights=False)[0]

    def forward(self, spec, wave=None, acoustic=None, wave90=None, az_k=0, return_aux=False, view_poses=None):
        B = spec.size(0); dev = spec.device; dir6 = self.dir6_buf.to(dev)
        t, poses, shifts, _ = self._encode(spec, az_k, view_poses)         # (B,nv,M,C)
        x = t.reshape(B * self.nv, self.M, self.C)
        for blk in self.view_blocks: x = blk(x)
        s = self._lift(x, dir6).reshape(B, self.nv, self.M, self.C)
        if self.variant != "noalign":                                      # deterministic azimuth roll
            outs = [torch.roll(s[:, i].transpose(1, 2).reshape(B, self.C, self.lh, self.lw),
                               shifts=int(shifts[i].item()), dims=-1).flatten(2).transpose(1, 2)
                    for i in range(self.nv)]
            s = torch.stack(outs, 1)
        if self.variant == "align_attn":
            local_dirs = torch.stack([_yaw_rot_inv(dir6[:, :3], y) for y, _ in poses], 0)
            Fm = self.view_fuse(s, local_dirs)
        else:
            Fm = s.mean(1)
        Fm = Fm + self.glob_dir(self._dirfeat(dir6)).unsqueeze(0)
        for blk in self.erp: Fm = blk(Fm)
        return self._decode(Fm, return_aux)


class OAAv2Depth(_OAABase):
    """v2: alternating intra/inter-mic attention + ray<->mic relative-geometry bias (incl. ear axis)."""
    def __init__(self, C=256, rounds=2, shuffle_pose=False, in_ch=1, norm="group", roll_sign=1,
                 lh=LH, lw=LW, dec_deep=False, enc_res=(128, 256), nviews=4, pe_mode="raw", pe_K=4, cond_mode="add",
                 multi_scale_lift=False, **hkw):
        super().__init__(C, in_ch, norm, roll_sign, lh, lw, dec_deep, enc_res, nviews, pe_mode, pe_K,
                         multi_scale_lift, **hkw)
        assert cond_mode in ("add", "adaln"), cond_mode
        self.shuffle_pose = shuffle_pose; self.cond_mode = cond_mode
        self.intra = nn.ModuleList([(CondSelfAttn(C) if cond_mode == "adaln" else SelfAttn(C)) for _ in range(rounds)])
        self.inter = nn.ModuleList([InterMicAttn(C) for _ in range(rounds)])
        self.ray_mic = RayMicAttn(C, pe_mode=pe_mode, pe_K=pe_K)

    def forward(self, spec, wave=None, acoustic=None, wave90=None, az_k=0, return_aux=False, view_poses=None):
        B = spec.size(0); dev = spec.device; dir6 = self.dir6_buf.to(dev); dir3 = dir6[:, :3]
        F4, poses, shifts, fine = self._encode(spec, az_k, view_poses)     # (B,nv,M,C), fine=(B,nv,4M,fine_ch)
        cond_bn = None
        if self.cond_mode == "adaln":                                      # B2: per-view pose_emb as AdaLN cond
            pf = torch.tensor([[math.sin(y), math.cos(y), e] for y, e in poses], device=dev)
            cond_bn = self.pose_emb(pf).unsqueeze(0).expand(B, -1, -1).reshape(B * self.nv, self.C)
        for intra, inter in zip(self.intra, self.inter):
            B_, N, M, C = F4.shape
            xin = F4.reshape(B_ * N, M, C)
            F4 = (intra(xin, cond_bn) if self.cond_mode == "adaln" else intra(xin)).reshape(B_, N, M, C)
            F4 = inter(F4)
        tokens = F4.reshape(B, self.nv * self.M, self.C)
        q = (self.q + self.dir_mlp(self._dirfeat(dir6)).unsqueeze(0)).expand(B, -1, -1)
        h = self.ray_mic(q, tokens, dir3, poses, self.M)
        if self.tof_mode == "pool": h = self._tof(h, F4)                  # C(pool, DEAD per T0): spec-token ToF
        elif self.tof_mode == "echo" and wave is not None: h = self._echo(h, wave)   # C(echo): waveform matched-filter
        elif self.tof_mode == "gcc" and wave is not None: h = self._gcc(h, wave)     # GCC-PHAT interaural timing
        h = h + self.glob_dir(self._dirfeat(dir6)).unsqueeze(0)
        for blk in self.erp: h = blk(h)
        if self.multi_scale_lift:
            return self._decode_ms(h, fine, poses, return_aux)
        return self._decode(h, return_aux)
