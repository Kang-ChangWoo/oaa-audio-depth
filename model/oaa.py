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
  value_moments  add a zero-initialised, per-channel multiple of the attention-weighted value
               standard deviation to its mean. +C parameters; global acoustic softmax only.
               This preserves a second statistic of the echo tokens, not an uncertainty score.

Architecture repairs (`archfix_` campaign, 2026-10-16; construction-time, default off => reference):
  fine_layerscale  LayerScale (CaiT) on the fine decoder skip: `fine_to_dec` keeps nn.Conv2d's own
               init and its output is multiplied by a learnable per-channel gamma initialised at
               this value. Replaces the zero-init, whose gradient to the whole fine branch is
               exactly W^T g = 0 at step 0. 0.0 = off (zero-init, the reference).
  registers    RayMicAttn register tokens (Darcet et al.): `registers` learnable C-vectors appended
               to the cross-attention keys/values, giving the softmax a place to put mass when no
               acoustic token is worth attending to. 0 = off. "global" fusion only.
  scene_registers  the registers additionally carry a pooled scene code,
               `reg + scene_mlp(tokens.mean(over tokens))`, with `scene_mlp` zero-initialised so the
               configuration starts exactly at plain `registers`. Requires registers > 0.
  geo_basis    "raw" (reference: the 5 geometry numbers straight into `bias_mlp`) | "sh4" (the
               rotated ray direction expanded in the real spherical harmonics of degree <= 4, 25
               coefficients, plus the 2 remaining scalars => 27 inputs). A fixed basis: it adds no
               parameters of its own, only the wider first `bias_mlp` layer.

Mid-band structure variants (2026-10-17; construction-time, default off => reference model). Both
target the 3-6 m band, which carries 11.1 % of the Replica pixel weight and 36-51 % of the weighted
squared error, and whose ray-grid resolution oracle (0.3077) explains only 42 % of our 0.7283:
  pose_reinject  the per-observation pose code is re-added before each `InterMicAttn` block,
               through one zero-initialised per-channel gamma per fusion round. The reference
               injects the pose code ONCE, into the coarse tokens in `_encode`; after that the only
               pose pathway through the fusion stack is `CondSelfAttn`'s AdaLN, and the
               inter-observation block -- the one site where the N observations are compared
               against each other, and therefore where the ear identity matters most -- has no pose
               conditioning at all. `--no-pose-emb` alone costs RMSE +12.56 % and mid +19.46 %,
               three times any other switch, and every pose experiment so far has REMOVED pose;
               this is the first one that adds it. gamma = 0 at init => the added term is exactly
               0.0, so the switch is bit-identical when turned on.
               NOT included, and the reason is measured: the fine ray-lift query is pose-blind too
               (one shared tensor broadcast over every observation), but any parameter placed there
               sits behind the zero-initialised `fine_to_dec` conv, whose gradient to the whole fine
               branch is exactly W^T g = 0 at step 0 -- the cold start `fine_layerscale` exists to
               repair. A gamma there measured a dead gradient at step 0 at N = 2/4/8 and was
               dropped rather than shipped unvalidated; it only becomes testable together with
               `--fine-layerscale`.
  geo_ray_gain   a per-(ray, head) multiplicative gain on the cross-attention OUTPUT, predicted
               OUTSIDE the softmax from [mean over observations of the geometry features,
               ray direction (3)] by a single zero-initialised `nn.Linear`, read as `1 + Linear(x)`.
               This is the only pathway in the model that can change the TOTAL magnitude a ray
               reads out. Distinct from its three relatives: the pre-softmax geometry bias can only
               redistribute a mass pinned to 1; GeoFiLM scales the post-softmax probabilities
               per (ray, OBSERVATION, head), so it both reweights observations against each other
               and changes the total; this gain is applied after the sum over observations and
               tokens, so it can change ONLY the total and cannot redistribute anything. `ray_gate`
               sits in the same position but is driven by `q_ln` (a per-ray query prior), and it sat
               flat at 1.0 in 13 of 14 runs; this one is driven by geometry. Deliberately NOT driven
               by logsumexp: the measured per-band mean lse is 10.179 / 10.201 / 10.146
               (near/mid/far) against band errors of 0.160 / 1.090 / 1.978, so lse is flat where the
               error is 12.4x apart and would fail as a distance sensor by construction.
Optional runtime argument (no parameters, no effect when omitted):
  forward(..., obs_valid=(B,N) bool)   explicit observation-validity masking of the inter-observation
               keys, the ray-cross keys/values and the fine-lift mean.
Decoder / target-side switches (construction-time or trainer-side, default off => reference model):
  erp_circular  decoder convs wrap the azimuth axis (ERP seam) instead of zero-padding it; no new
                parameters, identical state_dict keys
  yaw_roll_target(...)  trainer helper: roll the ERP target by k columns and shift the declared
                receiver yaws by the same angle (exact symmetry; used by train_oaa --yaw-roll-aug)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LH, LW = 16, 32                      # panoramic query grid (elevation x azimuth)

VIEW_FUSIONS = ("global", "hier", "hier-geo")
RAY_GATE_MODES = ("scalar", "channel")
GEO_BASES = ("raw", "sh4")
# geometry-bias input width per basis: raw = [R(-yaw)r (3), r.ear_axis*ear (1), ear (1)];
# sh4 = the same with the 3 direction components replaced by 25 real SH coefficients.
GEO_FEAT_DIM = {"raw": 5, "sh4": 27}
# feed-forward nonlinearity in the 4 `erp` query-refinement blocks. "off" is the reference
# LayerNorm -> Linear(C,2C) -> GELU -> Linear(2C,C).
SWIGLU_MODES = ("off", "wide", "matched")
# parameters that are deliberately initialised at zero/identity. They all share one measured
# symptom: after 40 epochs they have barely left their initial value (logit_scale 3-11 %,
# fine_to_dec RMS 0.0057 against the decoder's own 0.0173). `--param-lr-mult` gives exactly this
# set a different learning rate, which tests whether they are LR-starved rather than useless.
# `fine_ls` is deliberately NOT in this set: a LayerScale gamma is small but non-zero by design --
# starting non-zero is the whole point of it -- so it is not LR-starved in the way these are.
IDENTITY_INIT_PARAM_SUFFIXES = ("ray_mic.logit_scale", "ray_mic.geo_scale",
                                "ray_mic.gate_proj.weight", "ray_mic.gate_proj.bias",
                                "ray_mic.route_mlp.2.weight", "ray_mic.route_mlp.2.bias",
                                "ray_mic.scene_mlp.weight", "ray_mic.scene_mlp.bias",
                                "fine_to_dec.weight", "fine_to_dec.bias")
_QK_LOGSCALE_CLIP = 4.0              # exp() multiplier bounded to [1.8e-2, 5.5e1]
_QK_EPS = 1e-6
# ---- time-of-flight constants for --tof-value-code / --tof-tau-inject (2026-10-17).
# data_0422.py:14-15,32: 48 kHz, 279.9 samples per metre of ROUND TRIP, input window 2799 samples
# = exactly 10 m. The (257,T) spectrogram is resized to (256,512) and the encoder halves it four
# times, so the coarse token grid is 16 (frequency) x 32 (time) and ONE TIME COLUMN IS
# 10/32 = 0.3125 m. `h.flatten(2)` is row-major over (lh, lw), so token m sits in time column
# m % lw. That identity -- token time axis == depth axis -- is the whole premise of both switches.
_TOF_RANGE_M = 10.0
# Fourier ladder for the scalar arrival time. Log-spaced in cycles per metre, from half a cycle
# across the full 10 m window up to one cycle per 0.625 m = two time columns, i.e. the Nyquist
# limit of the 0.3125 m column grid. Going above that would encode a resolution the tokens do not
# carry. Closed form (not torch.logspace) so the buffer is identical on every machine.
_TOF_FOURIER_K = 16
_TOF_F_MIN, _TOF_F_MAX = 0.05, 1.6
# Tests flip this to False to exercise the masked code path on all-valid input; in production the
# short-circuit is what makes "obs_valid all True" bit-identical to "obs_valid=None".
_SHORTCIRCUIT_ALL_VALID = True
# Tests flip this to True to drive every register logit to -inf, which is how "registers present but
# unused must be bit-identical to no registers at all" is checked. Never set in production.
_MASK_REGISTERS = False


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


def _attention_value_std(prob, values, kv_keep=None):
    """Per-ray weighted value dispersion, with FP32 arithmetic under mixed precision.

    prob: (B,h,R,K), values: (B,h,K,dk). The reference mean/readout is NOT recomputed
    or changed. This separate statistic normalises the rounded probabilities, centres values
    on a valid token to reduce cancellation, and needs two additional value matmuls.
    No (B,h,R,K,dk) tensor is materialised. Subtracting sqrt(eps) makes constant values
    produce zero, while eps keeps the derivative finite at zero variance.
    """
    with torch.autocast(device_type=values.device.type, enabled=False):
        p, v = _hiprec(prob), _hiprec(values)
        if kv_keep is None:
            anchor = v[..., :1, :]
        else:
            # Choose the first VALID token, so changing a masked value cannot change the
            # numerical centring (including when observation zero is the one dropped).
            first = kv_keep.reshape(v.shape[0], -1).to(torch.int64).argmax(-1)
            index = first[:, None, None, None].expand(-1, v.shape[1], 1, v.shape[-1])
            anchor = v.gather(2, index)
        centred = v - anchor
        mass = p.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
        mean = (p @ centred) / mass
        variance = ((p @ centred.square()) / mass - mean.square()).clamp_min(0)
        eps = 1e-6
        return (variance + eps).sqrt() - math.sqrt(eps)

# pose (yaw, ear_sign) of each channel for the fixed input modes
_ALL_POSES = [(0.0, -1.0), (0.0, 1.0), (math.pi / 2, 1.0), (-math.pi / 2, -1.0)]   # [0L, 0R, 90R, 270L]
_POOL8 = [(0.0, -1.0), (0.0, 1.0), (math.pi / 2, -1.0), (math.pi / 2, 1.0),
          (math.pi, -1.0), (math.pi, 1.0), (-math.pi / 2, -1.0), (-math.pi / 2, 1.0)]   # [0L,0R,90L,90R,180L,180R,270L,270R]


def _tof_freqs(k):
    """k log-spaced spatial frequencies in cycles per metre, _TOF_F_MIN -> _TOF_F_MAX, float64.

    Computed in closed form rather than with torch.logspace so the value is reproducible across
    torch builds; the caller casts it to the module dtype.
    """
    i = torch.arange(k, dtype=torch.float64)
    return _TOF_F_MIN * (_TOF_F_MAX / _TOF_F_MIN) ** (i / max(k - 1, 1))


def _tof_column_metres(lh, lw):
    """Round-trip arrival time, in metres, of the time column each of the lh*lw tokens sits in.

    Token m -> column m % lw -> centre (m % lw + 0.5) * (_TOF_RANGE_M / lw). float64.
    """
    col = torch.arange(lh * lw, dtype=torch.float64) % lw
    return (col + 0.5) * (_TOF_RANGE_M / lw)


def _tof_sincos(tau_m, width):
    """Fixed sin/cos code of the per-token arrival time -> (len(tau_m), width), float64.

    width//2 frequencies, sines first then cosines. No parameters: this is a positional code of
    the DEPTH axis, deliberately content-independent, so what the readout learns is only how much
    of it to add (the per-channel gain), never what "4 metres" looks like.
    """
    assert width % 2 == 0, f"tof code width {width} must be even (sin/cos pairs)"
    f = _tof_freqs(width // 2)                                   # (width/2,) cycles per metre
    ang = 2 * math.pi * tau_m.unsqueeze(-1) * f.unsqueeze(0)     # (T, width/2)
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


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


# Real orthonormal spherical harmonics of degree <= 4 in Cartesian form, m = -l .. +l within each
# degree. The degree-3/4 rows use x^2+y^2+z^2 = 1 to drop the radial factors, so this is the basis
# *restricted to the unit sphere* -- which is exactly where it is evaluated: the rays come from
# `_dir_pe` (unit by construction) and `_yaw_rot_inv` is a rotation. Fixed basis, no parameters.
# Verified against scipy.special.sph_harm_y and for orthonormality in tests/test_oaa_arch_fixes.py.
_SH_C0 = 0.28209479177387814
_SH_C1 = 0.48860251190291987
_SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396)
_SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
          -0.4570457994644658, 1.445305721320277, -0.5900435899266435)
_SH_C4 = (2.5033429417967046, -1.7701307697799304, 0.9461746957575601, -0.6690465435572892,
          0.10578554691520431, -0.6690465435572892, 0.47308734787878004, -1.7701307697799304,
          0.6258357354491761)
_SH4_DIM = 25


def _sh4_basis(v):
    """Real SH up to degree 4 of unit vectors v (...,3) -> (...,25).

    Why a fixed basis instead of more MLP capacity: `bias_mlp` is a 2-layer MLP being asked to learn
    sin/cos-like angular functions from raw direction components, which is the spectral-bias failure
    that positional encoding fixes in NeRF. Being fixed, it costs no parameters of its own.

    Sign convention: these are the 3D-Gaussian-Splatting / plenoxels constants, which equal the
    canonical real SH times (-1)^|m|. The factor is a gauge choice -- the `bias_mlp` Linear that
    consumes this basis absorbs a per-column sign -- so span, orthonormality and the training
    trajectory are unchanged. `tests/test_oaa_arch_fixes.py::_SH4_SIGN` states the convention and
    checks it against scipy.
    """
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    o = torch.ones_like(x)
    return torch.stack([
        _SH_C0 * o,                                                              # l=0
        -_SH_C1 * y, _SH_C1 * z, -_SH_C1 * x,                                    # l=1
        _SH_C2[0] * xy, _SH_C2[1] * yz, _SH_C2[2] * (2.0 * zz - xx - yy),
        _SH_C2[3] * xz, _SH_C2[4] * (xx - yy),                                   # l=2
        _SH_C3[0] * y * (3.0 * xx - yy), _SH_C3[1] * xy * z,
        _SH_C3[2] * y * (4.0 * zz - xx - yy), _SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        _SH_C3[4] * x * (4.0 * zz - xx - yy), _SH_C3[5] * z * (xx - yy),
        _SH_C3[6] * x * (xx - 3.0 * yy),                                         # l=3
        _SH_C4[0] * xy * (xx - yy), _SH_C4[1] * yz * (3.0 * xx - yy),
        _SH_C4[2] * xy * (7.0 * zz - 1.0), _SH_C4[3] * yz * (7.0 * zz - 3.0),
        _SH_C4[4] * (35.0 * zz * zz - 30.0 * zz + 3.0),
        _SH_C4[5] * xz * (7.0 * zz - 3.0), _SH_C4[6] * (xx - yy) * (7.0 * zz - 1.0),
        _SH_C4[7] * xz * (xx - 3.0 * yy),
        _SH_C4[8] * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)),               # l=4
    ], -1)


def _norm(co, kind):
    return nn.GroupNorm(8, co) if kind == "group" else nn.BatchNorm2d(co)


def _padw(x):
    """Wrap one column on each side: azimuth -pi and +pi are the same direction on an ERP map."""
    return F.pad(x, (1, 1, 0, 0), mode="circular")


class _CircConv2d(nn.Conv2d):
    """3x3 conv whose azimuth (W) padding is circular and whose elevation (H) padding stays zero.
    Subclass, not wrapper: the state_dict keys are exactly those of the plain nn.Conv2d."""
    def __init__(self, ci, co, k, s=1, p=1):
        super().__init__(ci, co, k, s, (p, 0))

    def forward(self, x):
        return super().forward(_padw(x))


class _CircConvT2d(nn.ConvTranspose2d):
    """ConvTranspose2d(4, 2, 1) with a circular azimuth axis: wrap the input by one column, run the
    transposed conv, and drop the two output columns that each wrapped column produced. Tap-for-tap
    the same weights reach the same outputs as the zero-padded version, except that output columns at
    the seam now also see their true neighbours across the wrap."""
    def forward(self, x):
        return super().forward(_padw(x))[..., 2:-2]


class _ResBlk(nn.Module):
    def __init__(self, ci, co, down=True, norm="group", circular=False, stride=None):
        super().__init__()
        # `stride` overrides the (down -> 2, else 1) default with an explicit (freq, time) pair; used
        # only by ViewEncoder's --enc-time-stride1 path. Left None, s is the reference int.
        s = stride if stride is not None else (2 if down else 1)
        conv = _CircConv2d if circular else nn.Conv2d
        self.c1 = conv(ci, co, 3, s, 1); self.n1 = _norm(co, norm)
        self.c2 = conv(co, co, 3, 1, 1); self.n2 = _norm(co, norm)
        self.sc = nn.Conv2d(ci, co, 1, s, 0) if (ci != co or down) else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x))); h = self.n2(self.c2(h))
        return F.gelu(h + self.sc(x))


class ViewEncoder(nn.Module):
    """Shared per-observation encoder: strided conv stem + residual downsampling to (lh, lw) tokens.
    Also taps the (2lh, 2lw) feature map ("fine" tokens) for the multi-scale decoder skip."""
    def __init__(self, C=256, ngf=64, in_ch=1, norm="group", lh=LH, lw=LW, enc_res=(256, 512),
                 stem_stride1=False, time_stride1_stages=0):
        super().__init__()
        self.enc_res = enc_res; self.stem_stride1 = stem_stride1
        # time_stride1_stages (H1, 2026-10-18): the LAST k downsampling stages stride (2,1) instead of
        # (2,2), i.e. they halve frequency but keep the time (= arrival-distance) axis. The token
        # grid and the fine tap keep their reference shapes: the time axis is average-pooled back
        # to 2*lw at the fine tap and to lw before the tokens (exact power-of-two means, no
        # parameters). k=0 is the reference encoder, module for module; nothing below is touched.
        # Why: the reference folds time 4x after the stem (hop-16 STFT frames 175 -> 512 px -> 32
        # columns = 0.31 m per column); the two largest measured gains of this series (hop 160->16,
        # stem stride 2->1) are both input-time-resolution changes, so the hypothesis is that the
        # last two folds are where early-reflection evidence is lost. k=2 keeps 128 time columns
        # (0.078 m) through stages 3 and 4 and pools only at the interfaces.
        self.time_stride1_stages = int(time_stride1_stages)
        # stem_stride1: stride-1 stem on the native-resolution input instead of 2x-upsample + stride-2
        # (mathematically equivalent receptive field, no interpolation, 1/4 compute)
        self.stem = nn.Sequential(nn.Conv2d(in_ch, ngf // 2, 3, (1 if stem_stride1 else 2), 1), nn.GELU(),
                                  nn.Conv2d(ngf // 2, ngf, 3, 1, 1))
        stages = int(round(math.log2(enc_res[0] / lh)))                             # post-stem downsamplings
        assert enc_res[0] // lh == enc_res[1] // lw == 2 ** stages, f"enc_res {enc_res} -> {lh}x{lw}"
        assert stages >= 2, "multi-scale decoder needs >=2 encoder stages (fine tap at (2lh,2lw))"
        chans = [ngf, ngf * 2, ngf * 4, C, C, C][:stages] + [C]
        assert 0 <= self.time_stride1_stages <= stages, f"time_stride1_stages {time_stride1_stages} > {stages} stages"
        blocks = []
        for i in range(stages):
            keep_t = i >= stages - self.time_stride1_stages                            # last k stages keep time
            blocks += [_ResBlk(chans[i], chans[i], down=False, norm=norm),
                       _ResBlk(chans[i], chans[i + 1], norm=norm, stride=((2, 1) if keep_t else None))]
        self.net = nn.Sequential(*blocks); self.C = C
        self.lh, self.lw = lh, lw
        self.stages = stages
        self.fine_ch = chans[stages - 1]                                             # channels at the (2lh,2lw) tap

    def forward(self, x):
        want = tuple(self.enc_res) if self.stem_stride1 else tuple(2 * r for r in self.enc_res)
        if x.shape[-2:] != want:
            x = F.interpolate(x, size=want, mode="bilinear", align_corners=False)
        h = self.stem(x); ft = None
        if self.time_stride1_stages == 0:                                            # reference path, unchanged
            for blk in self.net:
                h = blk(h)
                if h.shape[-2:] == (2 * self.lh, 2 * self.lw):                       # fine tap
                    ft = h.flatten(2).transpose(1, 2)                                # (B, 4*lh*lw, fine_ch)
            tok = h.flatten(2).transpose(1, 2)                                       # (B, lh*lw, C)
            return tok, ft
        # --enc-time-stride1: the stage index decides the tap (shapes no longer identify it), and the
        # time axis is mean-pooled to the reference width at the two interfaces only.
        for i, blk in enumerate(self.net):
            h = blk(h)
            if i == 2 * (self.stages - 1) - 1:                                       # end of stage (stages-1): fine tap
                assert h.shape[-2] == 2 * self.lh, f"fine tap height {h.shape[-2]} != {2 * self.lh}"
                ft = self._pool_time(h, 2 * self.lw).flatten(2).transpose(1, 2)      # (B, 4*lh*lw, fine_ch)
        assert h.shape[-2] == self.lh, f"token height {h.shape[-2]} != {self.lh}"
        tok = self._pool_time(h, self.lw).flatten(2).transpose(1, 2)                 # (B, lh*lw, C)
        return tok, ft

    @staticmethod
    def _pool_time(h, w_out):
        k = h.shape[-1] // w_out
        assert k * w_out == h.shape[-1], f"time width {h.shape[-1]} not a multiple of {w_out}"
        return h if k == 1 else F.avg_pool2d(h, (1, k), stride=(1, k))


def _swiglu_hidden(C, mode):
    """Hidden width of the SwiGLU feed-forward, per mode.

    The reference block is LN(C) + Linear(C,2C) + Linear(2C,C) = 4C^2 + 5C parameters.
    A SwiGLU block is LN(C) + 2 x Linear(C,h) + Linear(h,C) = 3Ch + 2h + 3C.

    "wide"     h = 2C. The hidden width is untouched and the gating projection is simply added, so
               the block gains 2C^2 + 2C (131,584 at C=256; x4 blocks = +0.526 M). This BREAKS the
               15.45 M ceiling, which Changwoo approved explicitly on 2026-10-16.
    "matched"  the largest h whose SwiGLU parameter count does not EXCEED the reference block's:
               3Ch + 2h + 3C <= 4C^2 + 5C  =>  h <= (4C^2 + 2C)/(3C + 2). At C=256 that is 341.1,
               so h = 341 and the block lands 86 parameters UNDER the reference. The point of this
               version is "same parameter count, different nonlinearity", not more capacity.
    """
    assert mode in SWIGLU_MODES, f"swiglu mode {mode} not in {SWIGLU_MODES}"
    if mode == "wide":
        return 2 * C
    return (4 * C * C + 2 * C) // (3 * C + 2)


class _SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward, drop-in for the pre-LN `LayerNorm -> Linear -> GELU -> Linear` block:
    down(silu(gate(h)) * up(h)) instead of W2 GELU(W1 h).

    Unlike every archfix switch this CANNOT be bit-identical to the reference at any
    initialisation -- the nonlinearity itself is multiplicative and no zero/identity setting
    recovers GELU. So the gate is the two-part form B1 used: the flag OFF is bit-identical and
    strict-loads old checkpoints, and the flag ON is checked for a sane forward plus a live
    gradient on every tensor (tests/test_oaa_ffn_swiglu.py).
    """
    def __init__(self, C, hidden):
        super().__init__()
        self.hidden = int(hidden)
        self.n = nn.LayerNorm(C)
        self.gate = nn.Linear(C, self.hidden)
        self.up = nn.Linear(C, self.hidden)
        self.down = nn.Linear(self.hidden, C)

    def forward(self, x):
        h = self.n(x)
        return self.down(F.silu(self.gate(h)) * self.up(h))


class SelfAttn(nn.Module):
    """Pre-LN self-attention block (query refinement)."""
    def __init__(self, C, heads=8, swiglu="off"):
        super().__init__()
        self.n = nn.LayerNorm(C); self.a = nn.MultiheadAttention(C, heads, batch_first=True)
        # "off" keeps the exact reference Sequential, so its state_dict keys (m.0 / m.1 / m.3) and
        # every value are untouched and old checkpoints load with strict=True.
        self.m = (_SwiGLUFFN(C, _swiglu_hidden(C, swiglu)) if swiglu != "off" else
                  nn.Sequential(nn.LayerNorm(C), nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C)))

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
                 view_fusion="global", logit_scale_init=0.0, geo_scale_init=0.0,
                 ray_gate_mode="scalar", ray_gate_init=0.0,
                 registers=0, scene_registers=False, geo_basis="raw",
                 geo_value_film=False, film_lse=False, geo_ray_gain=False,
                 value_moments=False, tof_value_code=False, tof_tau_inject=False,
                 geo_key_coupling=False, tok_lh=LH, tok_lw=LW):
        super().__init__()
        assert view_fusion in VIEW_FUSIONS, f"view_fusion {view_fusion} not in {VIEW_FUSIONS}"
        assert ray_gate_mode in RAY_GATE_MODES, f"ray_gate_mode {ray_gate_mode} not in {RAY_GATE_MODES}"
        assert geo_basis in GEO_BASES, f"geo_basis {geo_basis} not in {GEO_BASES}"
        registers = int(registers)
        assert registers >= 0, f"registers must be >= 0, got {registers}"
        if scene_registers:
            assert registers > 0, ("--scene-registers conditions the register tokens on a pooled "
                                   "scene code, so it needs --registers > 0; refusing the "
                                   "combination instead of silently doing nothing")
        if registers:
            # under "hier" the outer softmax is over observations and a register belongs to no
            # observation, so there is no defined place to put it. Deliberately out of scope.
            assert view_fusion == "global", (
                f"--registers attaches to the single global softmax; view_fusion={view_fusion} "
                f"decomposes the softmax over observations and a register belongs to no "
                f"observation. Refusing the combination instead of guessing a semantics")
        self.h, self.dk = heads, C // heads
        self.value_moments = bool(value_moments)
        if self.value_moments:
            if (view_fusion != "global" or registers or geo_value_film or film_lse
                    or ray_gate or geo_ray_gain):
                raise ValueError("--value-moments requires the global acoustic softmax without "
                                 "registers, GeoFiLM, film-lse, ray-gate or geo-ray-gain; "
                                 "test one readout change at a time")
            # No RNG draw: same-seed shared weights AND the following training RNG stream stay
            # identical. A single zero-initialised scale has a live gradient at identity.
            self.moment_scale = nn.Parameter(torch.zeros(heads, C // heads))
        self.use_bias = use_bias          # False = "w/o geometric bias" ablation (content-only QK)
        self.nq = nn.LayerNorm(C); self.nk = nn.LayerNorm(C)
        self.q = nn.Linear(C, C); self.k = nn.Linear(C, C); self.v = nn.Linear(C, C); self.o = nn.Linear(C, C)
        self.geo_basis = geo_basis
        gf = GEO_FEAT_DIM[geo_basis]
        self.bias_mlp = nn.Sequential(nn.Linear(gf, 64), nn.GELU(), nn.Linear(64, heads))
        self.ffn = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))
        # ---- variants. Each one declares parameters ONLY when it is on, so the default model's
        # state_dict is byte-for-byte the reference one and old checkpoints load with strict=True.
        self.qk_norm, self.ray_gate, self.view_fusion = qk_norm, ray_gate, view_fusion
        self.ray_gate_mode = ray_gate_mode
        self.registers, self.scene_registers = registers, bool(scene_registers)
        self._pose_const = None           # (key, cos, sin, ear) cache; NOT state, see _pose_consts
        if registers:
            # same init as the panoramic queries. K/V reuse self.k / self.v, so the only new
            # parameters are the tokens themselves: registers * C.
            self.reg = nn.Parameter(torch.randn(1, registers, C) * 0.02)
        if scene_registers:
            # zero-initialised, so "scene-conditioned registers" starts EXACTLY at plain registers
            # and the two cells differ only in what the optimiser is allowed to learn
            self.scene_mlp = nn.Linear(C, C)
            nn.init.zeros_(self.scene_mlp.weight); nn.init.zeros_(self.scene_mlp.bias)
        if qk_norm:
            # log-scales. init 0 -> multiplier exactly 1 -> output identical to the no-qk-norm path.
            # The measured problem these inits address: after 40 epochs these parameters move only
            # 3-11 %, so whatever they are initialised to is effectively where they stay.
            self.logit_scale = nn.Parameter(torch.full((heads,), float(logit_scale_init)))
            self.geo_scale = nn.Parameter(torch.full((heads,), float(geo_scale_init)))
        if ray_gate:
            # scalar: one gate per (ray, head). channel: one gate per (ray, channel), the spec's
            # named follow-up. 2*sigmoid(bias) is the gate value at init, so bias 0 -> exactly 1.
            n_out = heads if ray_gate_mode == "scalar" else C
            self.gate_proj = nn.Linear(C, n_out)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, float(ray_gate_init))
        # ---- GeoFiLM (--geo-value-film). A per-(ray, observation, head) MULTIPLICATIVE
        # reweighting of the POST-softmax attention probabilities, driven by the same geometry
        # features the pre-softmax bias already uses. Distinct from all three existing relatives:
        #   * the geometry bias is pre-softmax, so it can only REDISTRIBUTE mass between tokens;
        #     total attention mass is pinned to 1. The output norm can still change through value
        #     selection/cancellation; what is absent is an independent mass/gain degree of freedom.
        #   * ray_gate is per-(ray, head), content-driven and geometry-blind, and is applied AFTER
        #     the sum over observations, so it cannot down-weight one observation against another.
        #   * registers are geometry-blind and spend their mass on a learned token.
        # Folding the scale into the PROBABILITIES (p' = p * film, then one p' @ V matmul) keeps the
        # reference reduction order, so film == 1.0 exactly at init is bit-identical, not merely
        # close. Scaling V per observation instead would split the value matmul and re-associate
        # the reduction -- the same thing that cost archfix C1 its bit-identity. Zero-init weight
        # AND bias => film = 1 + 0 = 1.0 exactly.
        self.geo_value_film = bool(geo_value_film)
        if self.geo_value_film:
            assert view_fusion == "global", (
                f"--geo-value-film reweights the single global softmax per observation; "
                f"view_fusion={view_fusion} already decomposes it per observation and the two "
                f"mechanisms would be fighting over the same quantity. Refusing the combination")
            assert use_bias, (
                "--geo-value-film reads the SAME geometry features as the pre-softmax bias; with "
                "--no-geo-bias those features are never computed. Refusing the combination")
            self.value_film = nn.Linear(gf, heads)
            nn.init.zeros_(self.value_film.weight)
            nn.init.zeros_(self.value_film.bias)
        # ---- C1 content abstention gate (--film-lse). ONE extra column on the film's input: the
        # per-(ray, observation, head) logsumexp of the pre-softmax logits over that observation's
        # M tokens, i.e. the total query-key match mass that the softmax is about to normalise
        # away. Softmax is invariant to a common ADDITIVE logit shift, not to logit scaling.
        # The unnormalised match mass is a content-dependent statistic, but is not by itself a
        # calibrated measure of audibility or uncertainty.
        #
        # Why this is NOT a retry of --ray-gate, which sat flat at 1.0 in 13 of 14 runs: ray_gate
        # is driven by q_ln, which is built from self.q and dir_mlp(dir6) only -- a per-ray prior
        # that never sees a key, so it had no input with which to detect "there is nothing here".
        # logsumexp(logits) is by construction a function of the key content. Same position in the
        # network, opposite driving signal. If this one also dies flat, that is the negative result
        # that closes the output-magnitude axis.
        #
        # Shape (heads,), deliberately NOT nn.Linear(1, heads): a bias column would duplicate
        # value_film.bias, which already carries the per-head constant. +8 parameters, not +16.
        # Zero-init => film_lse_w * lse is exactly 0.0, and film + 0.0 == film bit for bit, while
        # dfilm/dw = lse != 0 so the gradient is alive at step 0. The exponential null-token form
        # (softmax_1 / zero-value key), which is the literature's canonical abstention, cannot have
        # both: at the bias where keep == 1.0 exactly, dkeep/db is exactly -0.0 (measured).
        self.film_lse = bool(film_lse)
        if self.film_lse:
            assert self.geo_value_film, (
                "--film-lse adds a column to the GeoFiLM film input, so it requires "
                "--geo-value-film; refusing to build a gate with nothing to gate")
            self.film_lse_w = nn.Parameter(torch.zeros(heads))
        # ---- ray output-magnitude path (--geo-ray-gain). A per-(ray, head) scalar multiplying the
        # cross-attention OUTPUT, i.e. applied AFTER the sum over observations and tokens. It is the
        # only parameter in the model that can address the TOTAL magnitude a ray reads out:
        # `softmax(logits) @ V` normalises per ray, so a far ray with nothing to hear still emits a
        # full-magnitude vector. Measured symptom: attention output norm falls only 2.6 %
        # (near 4.433 -> far 4.319) while the band error rises 12.4x.
        #
        # Input is per-RAY, never per-observation -- that is exactly what separates it from GeoFiLM.
        # GeoFiLM multiplies the post-softmax probabilities per (ray, observation, head), so it
        # reweights the observations against each other AND changes the total; this one is applied
        # after that sum, so it changes the total and nothing else.
        #
        # The per-ray feature is [mean over observations of the geometry features (gf),
        # ray direction (3)]. The direction columns are not redundant: for a symmetric rig the
        # observation mean collapses almost everything. At r8 (yaws 0/90/180/270, each ear once) the
        # mean rotation matrix is exactly zero and the two ear columns cancel pairwise, so
        # `feats.mean(1)` is [0, dy, 0, 0, 0]; at r2 (both yaw 0, opposite ears) it is
        # [dx, dy, dz, 0, 0]. Without the raw direction the gain would have one live input at r8.
        #
        # The mean is over ALL observations, NOT a masked mean: the gain then does not read
        # obs_valid at all, which is what keeps `--subset-aug`'s masking invariance exact (the same
        # property GeoFiLM has). Zero weight AND zero bias => gain is exactly 1.0 and `x * 1.0` is
        # exact in IEEE-754, so turning the switch on moves no bit at init.
        self.geo_ray_gain = bool(geo_ray_gain)
        if self.geo_ray_gain:
            assert use_bias, (
                "--geo-ray-gain is driven by the SAME geometry features as the pre-softmax bias; "
                "with --no-geo-bias those features are never computed. Refusing the combination")
            self.ray_gain = nn.Linear(gf + 3, heads)
            nn.init.zeros_(self.ray_gain.weight)
            nn.init.zeros_(self.ray_gain.bias)
        if view_fusion == "hier-geo":
            self.route_mlp = nn.Sequential(nn.Linear(gf, 64), nn.GELU(), nn.Linear(64, heads))
            nn.init.zeros_(self.route_mlp[-1].weight); nn.init.zeros_(self.route_mlp[-1].bias)
        # ---- arrival-time readout switches (--tof-value-code, --tof-tau-inject, 2026-10-17).
        # Both read the SAME quantity the softmax already computed -- the attention mass per time
        # column -- and hand its first moment to the decoder as an explicit, content-independent
        # number. Nothing else in the model does: the time axis enters the values only through
        # `tf_pe`, which is a learnable code that LayerNorm, the v projection and two fusion rounds
        # have already mixed into the content by the time the readout sees it. A depth estimate as
        # simple as "the weighted mean arrival time" therefore has no clean path to the head.
        #
        # They are registered against the TOKEN grid (lh, lw), which the caller passes in; the
        # module cannot infer it, because M only arrives at forward time and lh*lw does not factor.
        self.tok_lh, self.tok_lw = int(tok_lh), int(tok_lw)
        self.tof_value_code = bool(tof_value_code)
        self.tof_tau_inject = bool(tof_tau_inject)
        if self.tof_value_code or self.tof_tau_inject:
            assert view_fusion == "global", (
                f"--tof-value-code / --tof-tau-inject read the post-softmax probabilities of the "
                f"single global softmax; view_fusion={view_fusion} decomposes it per observation "
                f"and never materialises them. Refusing the combination instead of guessing")
            # non-persistent: a fixed function of (lh, lw), so it must not enter state_dict and
            # cannot make an existing checkpoint fail strict loading
            self.register_buffer("tof_tau_m", _tof_column_metres(self.tok_lh, self.tok_lw).float(),
                                 persistent=False)
        if self.tof_value_code:
            # Fixed sin/cos code of the arrival time, dk wide, SHARED by the heads; only the gain
            # is learned, one per (head, channel) = +C. torch.zeros draws no RNG, so every later
            # init -- and therefore every shared weight at a given seed -- is unchanged, which is
            # what keeps this cell a pure mechanism comparison against its control seed.
            self.register_buffer("tof_code",
                                 _tof_sincos(_tof_column_metres(self.tok_lh, self.tok_lw),
                                             self.dk).float(), persistent=False)
            self.tof_gain = nn.Parameter(torch.zeros(heads, self.dk))
        if self.tof_tau_inject:
            # tau_bar, the attention-weighted arrival time, is ONE SCALAR PER RAY: a linear
            # estimator of depth. Fourier-lift it (a raw metre scalar against LayerNorm'd features
            # would be badly scaled) and send it to C with a single zero-init affine.
            #
            # Deliberately NO separate zero-init gamma on top, although the brief asked for one:
            # with both a zero gamma and a zero affine the affine's own gradient at step 0 is
            # gamma * (...) = exactly 0, i.e. the dead-branch failure mode that cost --pose-reinject
            # its fine site (test_g3b there). A zero-init affine ALONE is bit-identical at init
            # (0 @ x + 0 == 0.0) and has a live gradient (dL/dW = g (x) feat, dL/db = g), so it is
            # the form that satisfies both halves of the gate. Measured in test_g3c here.
            self.register_buffer("tau_freq", _tof_freqs(_TOF_FOURIER_K).float(), persistent=False)
            self.tau_w = nn.Parameter(torch.zeros(2 * _TOF_FOURIER_K, C))
            self.tau_b = nn.Parameter(torch.zeros(C))
        # ---- geometry-key coupling (--geo-key-coupling, 2026-10-17). The reference geometry bias
        # `bmic` is (R, N, h) and is COPIED across the M tokens of an observation (see the expand
        # below), so geometry can say "listen to this observation" but never "listen to this
        # frequency row / this time column". Every relative that has been tried -- GeoFiLM,
        # hier-geo routing, geo-ray-gain, the sh4 basis -- is the same per-(ray, observation, head)
        # scalar in a different costume. This one sends the geometry into the KEY space and takes
        # an inner product with the actual keys, so the term it adds to the logits is per TOKEN.
        #
        # Per head (R, N, h, dk), not head-shared: a head-shared map forces every head to look
        # along the same geometric direction in key space, which is the restriction the per-head
        # Q/K projections exist to avoid. Linear(gf+3, h*dk) without bias -- a bias would be a
        # per-head constant over all tokens, which is exactly what `bias_mlp` already provides.
        # The raw ray direction is concatenated for the same measured reason as --geo-ray-gain.
        self.geo_key_coupling = bool(geo_key_coupling)
        if self.geo_key_coupling:
            assert use_bias, (
                "--geo-key-coupling is driven by the SAME geometry features as the pre-softmax "
                "bias; with --no-geo-bias those features are never computed. Refusing the "
                "combination")
            self.geo_key = nn.Parameter(torch.zeros(gf + 3, C))
        self.collect_stats = False        # diagnostics only; never read by the loss
        self.stats = {}

    def _pose_consts(self, poses, device, dtype):
        """cos/sin of every yaw and every ear sign as (N,) tensors, cached on the pose list.

        The reference built `torch.tensor([cos, 0, -sin])` inside a per-observation Python loop, i.e.
        N host-to-device copies per forward (8 of them at 8 observations). PREREG called for a
        registered buffer; a buffer cannot be used, because `poses` is a *forward-time* argument
        (`view_poses=` overrides the construction-time list, and data_micgain passes the dataset's
        own ordering), so the constants are cached keyed on the pose list instead. Not state: never
        in state_dict, invalidated by any dtype/device/pose change.
        """
        key = (tuple((float(y), float(e)) for y, e in poses), str(device), dtype)
        if self._pose_const is None or self._pose_const[0] != key:
            t = lambda vals: torch.tensor(vals, device=device, dtype=dtype)
            self._pose_const = (key,
                                t([math.cos(float(y)) for y, _ in poses]),
                                t([math.sin(float(y)) for y, _ in poses]),
                                t([float(e) for _, e in poses]))
        return self._pose_const[1:]

    def _geo_feats(self, ray_dir3, poses):
        """Per-observation geometry features [R(-yaw) r (3), r.ear_axis * ear (1), ear (1)].

        -> (R, N, 5), batched over observations. Bit-identical to the per-observation loop it
        replaces, forward AND gradients (measured max|diff| = 0.0 at float32 and float64,
        N = 2/4/8, gated by tests/test_oaa_arch_fixes.py); the ear-axis column needs no matmul at
        all because
        r . (R(yaw) x_hat) = dx cos(yaw) - dz sin(yaw) is literally the first component of
        R(-yaw) r, which the rotation already computed.
        dtype follows ray_dir3 so the module also runs in float64; at float32 (and under bf16
        autocast, where ray_dir3 is the float32 dir buffer) these are the original float32 values.
        """
        cs, ss, ea = self._pose_consts(poses, ray_dir3.device, ray_dir3.dtype)
        dx, dy, dz = ray_dir3[..., 0:1], ray_dir3[..., 1:2], ray_dir3[..., 2:3]      # (R,1)
        lx = dx * cs - dz * ss                                                        # (R,N)
        lz = dx * ss + dz * cs
        ly = dy.expand_as(lx)
        return torch.stack([lx, ly, lz, lx * ea, ea.expand_as(lx)], -1)               # (R,N,5)

    def _geo_basis_feats(self, ray_dir3, poses):
        """The geometry-bias input in the configured basis: (R, N, GEO_FEAT_DIM[geo_basis])."""
        f = self._geo_feats(ray_dir3, poses)
        if self.geo_basis == "raw":
            return f
        return torch.cat([_sh4_basis(f[..., :3]), f[..., 3:]], -1)                    # (R,N,27)

    def _logscale(self, p):
        return p.clamp(-_QK_LOGSCALE_CLIP, _QK_LOGSCALE_CLIP).exp()

    def _register_kv(self, tokens, Q, Qn, B):
        """Register keys/values and their logits -> (V_reg (B,h,n_reg,dk), logits (B,h,R,n_reg)).

        The registers go through the SAME nk / k / v projections as the acoustic tokens, so they add
        no parameters beyond the tokens themselves and they live in the same key space the queries
        already search. Rationale (Darcet et al.): a far ray has nothing worth attending to, the
        softmax must still sum to one, and it dumps that mass on whatever token is least wrong --
        which is the artifact. A register is a learnable place to put it.
        """
        reg = self.reg
        if self.scene_registers:
            # pooled scene code, the global pathway the model otherwise has nowhere. Mean over the
            # acoustic tokens of the whole observation set, i.e. one code per sample.
            reg = reg + self.scene_mlp(tokens.mean(1, keepdim=True))       # (B,n_reg,C)
        rk = self.nk(reg.expand(B, -1, -1) if reg.shape[0] == 1 else reg)
        Kr = self.k(rk).view(B, -1, self.h, self.dk).transpose(1, 2)
        Vr = self.v(rk).view(B, -1, self.h, self.dk).transpose(1, 2)
        if self.qk_norm:
            creg = (Qn @ _rms_norm_heads(Kr).transpose(-2, -1)) / math.sqrt(self.dk)
            lreg = (creg * self._logscale(self.logit_scale).view(1, -1, 1, 1)).to(Q.dtype)
        else:
            lreg = (Q @ Kr.transpose(-2, -1)) / math.sqrt(self.dk)
        # no geometry bias on a register: it is attached to no observation and no direction
        if _MASK_REGISTERS:
            lreg = torch.full_like(lreg, float("-inf"))
        return Vr, lreg

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
            delta = self.route_mlp(feats)                                  # (R,N,h) from (R,N,gf)
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
        feats = self._geo_basis_feats(ray_dir3, poses) if (self.use_bias or self.view_fusion == "hier-geo") else None
        Qn = _rms_norm_heads(Q) if self.qk_norm else None
        if self.qk_norm:
            # decouple the content logit scale from ||Q||,||K|| so the geometry bias and the content
            # term keep a learnable, head-wise relative size instead of an accidental one
            content = (Qn @ _rms_norm_heads(K).transpose(-2, -1)) / math.sqrt(self.dk)
            logits = (content * self._logscale(self.logit_scale).view(1, -1, 1, 1)).to(Q.dtype)
        else:
            logits = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)
        if self.use_bias:
            # N calls on the per-observation slices, NOT one call on (R,N,gf). PREREG C1 asked for
            # the single call ("8 kernel launches per forward"), but it cannot ship: measured on
            # this node, the one-call form is bit-identical in the FORWARD and its
            # bias_mlp.2.weight gradient differs by 1.08e-18 relative to the largest gradient in
            # the configuration -- one wide gemm reduction over R*N rows instead of N accumulated
            # reductions over R. C1's gate is exact bit-identity, and the gate wins over the
            # launch count. The host-to-device copies, which were the expensive part, are gone
            # either way (see _pose_consts).
            bmic = torch.stack([self.bias_mlp(feats[:, i].contiguous()) for i in range(N)], 1)
            if self.qk_norm:
                bmic = bmic * self._logscale(self.geo_scale).view(1, 1, -1)
            bfull = bmic.unsqueeze(2).expand(R, N, M, self.h).reshape(R, N * M, self.h)
            logits = logits + bfull.permute(2, 0, 1).unsqueeze(0)
        if self.geo_key_coupling:
            # [geometry features (gf), raw ray direction (3)] -> (R, N, h, dk) -> <g, K> per token.
            # Zero weight => every element of `add` is a sum of exact zeros => exactly 0.0, and
            # `logits + 0.0` moves no bit. dL/dW = sum (dL/dlogit) f (x) K is non-zero because the
            # keys are not zero, so the one layer learns from step 0.
            gin = torch.cat([feats, ray_dir3.to(feats.dtype).unsqueeze(1).expand(R, N, 3)], -1)
            gk = (gin @ self.geo_key.to(feats.dtype)).view(R, N, self.h, self.dk)
            Kn = K.reshape(B, self.h, N, M, self.dk)
            add = torch.einsum("rnhd,bhnmd->bhrnm", gk.to(K.dtype), Kn)
            logits = logits + add.reshape(B, self.h, R, N * M)
        kv_keep = None
        if _mask_engaged(obs_valid):
            kv_keep = obs_valid[:, :, None].expand(B, N, M).reshape(B, 1, 1, N * M)
        Vr = lreg = None
        if self.registers:
            Vr, lreg = self._register_kv(tokens, Q, Qn, B)
        film = None
        if self.geo_value_film:
            # feats (R,N,gf) -> (R,N,h) -> (1,h,R,N), then kept as (1,h,R,N,1) and BROADCAST
            # against a free reshape of the probabilities. Materialising it at (1,h,R,N*M) would
            # allocate a second tensor the size of the logits for no benefit.
            fv = (1.0 + self.value_film(feats)).permute(2, 0, 1).unsqueeze(0)   # (1,h,R,N)
            film = fv.unsqueeze(-1).to(logits.dtype)                            # (1,h,R,N,1)
            if self.film_lse:
                # (B,h,R,N) off the very logits this branch is about to normalise. Reduce on a
                # RESHAPED VIEW, never on a materialised copy: logsumexp's output is (B,h,R,N),
                # which is M times smaller than the logits, so the only large allocation is the
                # masked_fill below and that happens only when an observation is actually dropped.
                # A fully dropped observation is all -inf -> logsumexp = -inf -> 0.0 * -inf = nan.
                # Its probabilities are exactly 0 either way, so non-finite rows are replaced by 0
                # instead of being allowed to poison the product.
                lg = logits.reshape(B, self.h, R, N, M)
                if kv_keep is not None:
                    lg = lg.masked_fill(~kv_keep.reshape(B, 1, 1, N, M), float("-inf"))
                lse = lg.logsumexp(-1)
                lse = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
                film = film + (self.film_lse_w.view(1, -1, 1, 1, 1)
                               * lse.unsqueeze(-1)).to(film.dtype)
            if self.collect_stats:
                with torch.no_grad():
                    # report the multiplier that is actually applied, not the geometry half of it,
                    # so M1/M3 read the same tensor the forward used. With --film-lse off this is
                    # fv to the bit, so the GeoFiLM gates are unaffected.
                    f32 = film.detach().squeeze(-1).float()
                    self.stats.update(film_mean=f32.mean().item(), film_std=f32.std().item(),
                                      film_min=f32.min().item(), film_max=f32.max().item())
                    self.stats["film_mean_per_view"] = f32.mean((0, 1, 2)).cpu()
        if self.view_fusion == "global":
            if kv_keep is None and lreg is None:
                p_tok = logits.softmax(-1)
                if film is not None:            # film == 1.0 exactly at init -> bitwise no-op
                    p_tok = (p_tok.reshape(B, self.h, R, N, M) * film).reshape(B, self.h, R, N * M)
                out_head = p_tok @ V                                       # reference path, unchanged
            elif lreg is None:                                             # FP32 softmax; dropped keys get weight 0
                p_tok = _hiprec(logits).masked_fill(~kv_keep, float("-inf")).softmax(-1).to(V.dtype)
                if film is not None:
                    p_tok = (p_tok.reshape(B, self.h, R, N, M) * film).reshape(B, self.h, R, N * M)
                out_head = p_tok @ V
            else:
                # one softmax over [acoustic tokens | registers], then the two value matmuls kept
                # SEPARATE. That is what makes "registers at -inf" bit-identical to no registers:
                # the token half is the same contiguous matmul as the reference and the register
                # half contributes exact zeros, instead of widening one reduction by 4 columns.
                lt = _hiprec(logits) if kv_keep is None else \
                    _hiprec(logits).masked_fill(~kv_keep, float("-inf"))
                p = torch.cat([lt, _hiprec(lreg)], -1).softmax(-1)
                p_tok = p[..., :N * M].contiguous().to(V.dtype)
                p_reg = p[..., N * M:].contiguous().to(V.dtype)
                if film is not None:            # registers are attached to no observation
                    p_tok = (p_tok.reshape(B, self.h, R, N, M) * film).reshape(B, self.h, R, N * M)
                out_head = p_tok @ V + p_reg @ Vr
                if self.collect_stats:
                    with torch.no_grad():
                        rm = p_reg.detach().float().sum(-1)                 # (B,h,R) mass on registers
                        self.stats["reg_mass_mean"] = rm.mean().item()
                        self.stats["reg_mass_max"] = rm.max().item()
                        self.stats["reg_mass_per_head"] = rm.mean((0, 2)).cpu()
        else:
            out_head = self._hier_fuse(logits, V, N, M, feats, kv_keep)
        if self.value_moments:
            dispersion = _attention_value_std(p_tok, V, kv_keep)
            correction = (self.moment_scale[None, :, None, :] * dispersion).to(out_head.dtype)
            out_head = out_head + correction
            if self.collect_stats:
                with torch.no_grad():
                    self.stats["value_std_mean"] = dispersion.mean().item()
                    self.stats["moment_scale_absmean"] = self.moment_scale.abs().mean().item()
                    self.stats["moment_correction_rms"] = correction.float().square().mean().sqrt().item()
        tau_term = None
        if self.tof_value_code or self.tof_tau_inject:
            # (B,h,R,N,M) view of the acoustic probabilities. Registers are excluded by
            # construction: `p_tok` is already the token half of the joint softmax, so a ray that
            # parks mass on a register contributes that much less here, which is the behaviour we
            # want (abstaining should shrink the arrival-time term, not bias it).
            p5 = p_tok.reshape(B, self.h, R, N, M)
            if self.collect_stats:
                with torch.no_grad():
                    # attention mass per TIME COLUMN, head- and observation-averaged: (B,R,lw).
                    # This is the P1 probe quantity (where on the depth axis a ray is listening)
                    # and it is what tau_bar is the first moment of.
                    cm = (p5.detach().float().sum((1, 3))
                          .reshape(B, R, self.tok_lh, self.tok_lw).sum(2) / self.h)
                    self.stats["tof_col_mass"] = cm.mean((0, 1)).cpu()
                    self.stats["tof_col_mass_full"] = cm.cpu()
        if self.tof_value_code:
            # p @ E: the attention-weighted fixed time code. Same shape of operation as p @ V, but
            # the "values" carry depth and nothing else. Per-(head, channel) zero-init gain, so the
            # added term is exactly 0.0 at init and dL/dgain = g * (p@E) is live.
            read = torch.einsum("bhrnm,md->bhrd", p5, self.tof_code.to(p5.dtype))
            out_head = out_head + (self.tof_gain.view(1, self.h, 1, self.dk).to(read.dtype)
                                   * read).to(out_head.dtype)
            if self.collect_stats:
                with torch.no_grad():
                    self.stats["tof_read_rms"] = read.float().square().mean().sqrt().item()
                    self.stats["tof_gain_absmean"] = self.tof_gain.abs().mean().item()
        if self.tof_tau_inject:
            # tau_bar (B,R) in metres: sum over heads / tokens of p * column centre, divided by the
            # head count. self.h is a power of two, so the division is exact.
            tau = torch.einsum("bhrnm,m->br", p5, self.tof_tau_m.to(p5.dtype)) / self.h
            ang = 2 * math.pi * tau.unsqueeze(-1) * self.tau_freq.to(tau.dtype).view(1, 1, -1)
            ff = torch.cat([torch.sin(ang), torch.cos(ang)], -1)            # (B,R,2K)
            tau_term = ff @ self.tau_w.to(ff.dtype) + self.tau_b.to(ff.dtype)
            if self.collect_stats:
                with torch.no_grad():
                    t32 = tau.detach().float()
                    self.stats.update(tau_bar_mean=t32.mean().item(), tau_bar_std=t32.std().item(),
                                      tau_bar_min=t32.min().item(), tau_bar_max=t32.max().item())
                    self.stats["tau_bar_per_ray"] = t32.mean(0).cpu()
        if self.geo_ray_gain:
            # (R,N,gf) -> (R,gf) over observations, concatenated with the raw ray direction, then
            # one Linear to (R,h) and a broadcast against (B,h,R,dk). Applied BEFORE ray_gate; the
            # two multipliers commute and `* 1.0` is exact, so the order cannot affect the gate.
            rf = torch.cat([feats.mean(1), ray_dir3.to(feats.dtype)], -1)       # (R, gf+3)
            rg = (1.0 + self.ray_gain(rf)).t().unsqueeze(0).unsqueeze(-1)       # (1,h,R,1)
            out_head = out_head * rg.to(out_head.dtype)
            if self.collect_stats:
                with torch.no_grad():
                    g32 = rg.detach().float().squeeze(0).squeeze(-1)            # (h,R)
                    self.stats.update(ray_gain_mean=g32.mean().item(), ray_gain_std=g32.std().item(),
                                      ray_gain_min=g32.min().item(), ray_gain_max=g32.max().item())
                    self.stats["ray_gain_mean_per_head"] = g32.mean(1).cpu()
                    # per-ray, head-averaged: the only shape from which "does the gain fall with
                    # distance" can be answered post hoc against the ERP ground truth
                    self.stats["ray_gain_per_ray"] = g32.mean(0).cpu()
        if self.ray_gate:
            gate = 2.0 * torch.sigmoid(self.gate_proj(q_ln))               # in (0,2); =1 at bias 0
            if self.ray_gate_mode == "scalar":
                g = gate.permute(0, 2, 1).unsqueeze(-1)                    # (B,R,h)  -> (B,h,R,1)
            else:
                g = gate.view(B, R, self.h, self.dk).transpose(1, 2)       # (B,R,C)  -> (B,h,R,dk)
            out_head = out_head * g
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
        h = h + self.ffn(h)
        if tau_term is not None:
            # added to the block OUTPUT, not to `out`: the point of the switch is to hand the
            # decoder a depth estimate that has not been through another projection and FFN.
            h = h + tau_term.to(h.dtype)
        return h


def _make_up(C, norm, lh, lw, deep=True, circular=False):
    """Decoder stages (lh,lw) -> (256,512), #stages = log2(256/lh); returns (ModuleList, per-stage out channels).
    circular=True wraps the azimuth axis in every decoder conv (same parameters, same keys)."""
    stages = int(round(math.log2(256 / lh)))
    assert 256 % lh == 0 and 512 % lw == 0 and (256 // lh) == (512 // lw), f"bad grid {lh}x{lw}"
    chans = [C, 128, 64, 32, 16, 16, 16][:stages + 1]
    convT = _CircConvT2d if circular else nn.ConvTranspose2d

    def up(ci, co):
        blks = [convT(ci, co, 4, 2, 1), _norm(co, norm), nn.GELU(), _ResBlk(co, co, down=False, norm=norm, circular=circular)]
        if deep: blks.append(_ResBlk(co, co, down=False, norm=norm, circular=circular))
        return nn.Sequential(*blks)
    return nn.ModuleList([up(chans[i], chans[i + 1]) for i in range(stages)]), chans[1:stages + 1]


def yaw_roll_target(gt, mask, poses, k, width=512):
    """Exact yaw symmetry of the ERP target: declaring every receiver `2*pi*k/width` further round
    (counter-clockwise, in the model's +A convention) is the same physical scene with the panoramic
    depth map rolled by +k columns. torch.roll(+k) moves column c to c+k, and `_fine_lift` rolls a
    receiver's own frame by +round(yaw/(2pi)*2lw) to reach the global frame, so both use the same sign.
    k=0 returns the inputs untouched. The audio is not changed: the same recording, re-labelled."""
    if k == 0:
        return gt, mask, poses
    dyaw = 2 * math.pi * k / width
    return (torch.roll(gt, k, dims=-1), torch.roll(mask, k, dims=-1),
            [(float(y) + dyaw, float(e)) for y, e in poses])


class OAAv2Depth(nn.Module):
    """Reference OAA model (the one used for all reported results)."""
    def __init__(self, C=256, rounds=2, in_ch=1, norm="group", lh=LH, lw=LW, dec_deep=True,
                 enc_res=(256, 512), nviews=4, stem_stride1=False, max_depth=10.0,
                 no_pose_emb=False, no_ray_emb=False, no_geo_bias=False, no_tf_pe=False, no_cross=False,
                 qk_norm=False, ray_gate=False, view_fusion="global",
                 logit_scale_init=0.0, geo_scale_init=0.0, ray_gate_mode="scalar",
                 ray_gate_init=0.0, fine_skip_init=0.0, erp_circular=False,
                 fine_layerscale=0.0, registers=0, scene_registers=False, geo_basis="raw",
                 geo_value_film=False, film_lse=False, ffn_swiglu="off",
                 pose_reinject=False, geo_ray_gain=False, value_moments=False,
                 tof_value_code=False, tof_tau_inject=False, cross_reread=False,
                 geo_key_coupling=False, enc_time_stride1=False):
        super().__init__()
        assert ffn_swiglu in SWIGLU_MODES, f"ffn_swiglu {ffn_swiglu} not in {SWIGLU_MODES}"
        if value_moments and pose_reinject:
            raise ValueError("--value-moments and --pose-reinject change different parts of the "
                             "fusion stack; test one structural change at a time")
        self.ffn_swiglu = ffn_swiglu
        # erp_circular: the decoder convs (ConvTranspose2d, both ResBlk 3x3 convs per stage, head)
        # wrap the azimuth axis instead of zero-padding it. ERP columns 0 and 511 are adjacent
        # directions (behind the listener); zero padding tells the decoder they are image borders.
        # No new parameters, identical state_dict keys; False reproduces the reference exactly.
        self.erp_circular = erp_circular
        # the attention/fusion variants all live inside RayMicAttn, which --no-cross removes entirely
        if no_cross:
            if value_moments:
                raise ValueError("--no-cross removes the --value-moments readout")
            assert not (qk_norm or ray_gate or view_fusion != "global"), (
                "--no-cross removes RayMicAttn, so --qk-norm / --ray-gate / --view-fusion have no "
                "effect; refusing the combination instead of silently ignoring it")
            assert not (int(registers) or scene_registers or geo_basis != "raw"), (
                "--no-cross removes RayMicAttn, so --registers / --scene-registers / --geo-basis "
                "have no effect; refusing the combination instead of silently ignoring it")
            assert not (geo_value_film or film_lse), (
                "--no-cross removes RayMicAttn, so --geo-value-film / --film-lse have no effect; "
                "refusing the combination instead of silently ignoring it")
            assert not geo_ray_gain, (
                "--no-cross removes RayMicAttn, so --geo-ray-gain has no output to scale; "
                "refusing the combination instead of silently ignoring it")
            assert not (tof_value_code or tof_tau_inject or geo_key_coupling or cross_reread), (
                "--no-cross removes RayMicAttn, so --tof-value-code / --tof-tau-inject / "
                "--geo-key-coupling / --cross-reread have no attention to read; refusing the "
                "combination instead of silently ignoring it")
        # pose re-injection and the pose ablation are the same pathway read in opposite directions
        assert not (pose_reinject and no_pose_emb), (
            "--pose-reinject re-states the pose code that --no-pose-emb removes; the two are the "
            "same pathway in opposite directions. Refusing the combination")
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
        # enc_time_stride1 (H1): the last two encoder stages keep the time axis; token grid, C and
        # the (2lh,2lw) fine tap are unchanged, so nothing downstream of the encoder knows.
        self.enc_time_stride1 = bool(enc_time_stride1)
        self.enc = ViewEncoder(C, in_ch=in_ch, norm=norm, lh=lh, lw=lw, enc_res=enc_res, stem_stride1=stem_stride1,
                               time_stride1_stages=(2 if enc_time_stride1 else 0))
        self.pose_emb = nn.Sequential(nn.Linear(3, C), nn.GELU(), nn.Linear(C, C))       # MLP_pose([sin, cos, ear])
        self.tf_pe = nn.Parameter(torch.zeros(1, self.M, C))                              # learnable TF PE
        self.q = nn.Parameter(torch.randn(1, self.M, C) * 0.02)                           # learnable panoramic queries
        self.dir_mlp = nn.Sequential(nn.Linear(6, C), nn.GELU(), nn.Linear(C, C))        # MLP_ray (query side)
        self.glob_dir = nn.Sequential(nn.Linear(6, C), nn.GELU(), nn.Linear(C, C))       # ray emb added after fusion
        self.erp = nn.ModuleList([SelfAttn(C, swiglu=ffn_swiglu) for _ in range(4)])      # 4 refinement blocks
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
        self.up_stages, up_ch = _make_up(C, norm, lh, lw, deep=dec_deep, circular=erp_circular)
        self.fine_in = nn.Linear(self.enc.fine_ch, C)
        self.fine_q = nn.Parameter(torch.randn(1, 4 * self.M, C) * 0.02)
        self.fine_tf_pe = nn.Parameter(torch.zeros(1, 4 * self.M, C))
        self.fine_dir_mlp = nn.Sequential(nn.Linear(6, C), nn.GELU(), nn.Linear(C, C))
        self.fine_ln = nn.LayerNorm(C)
        self.fine_lift = nn.MultiheadAttention(C, 8, batch_first=True)
        # zero-init by default: the fine ray-lift skip starts with no injection and has to learn its
        # way in. Measured after 40 epochs: weight RMS 0.0057 against the decoder's own first conv
        # at 0.0173, i.e. the only high-resolution path stays nearly dead. fine_skip_init > 0 gives
        # it a non-zero start instead; 0.0 reproduces the reference exactly.
        #
        # fine_layerscale > 0 is the archfix repair of the same symptom, and it is a different fix:
        # the zero-init does not merely start small, it makes the gradient reaching `fine_erp`
        # exactly W^T g = 0, so fine_q / fine_dir_mlp / fine_in / fine_lift get NO gradient at step
        # 0 and the branch cannot start learning until fine_to_dec drifts off zero on its own.
        # LayerScale keeps the conv at nn.Conv2d's own init (so the gradient path is alive from step
        # 0) and scales the branch output by a learnable per-channel gamma instead. fine_skip_init
        # addressed the magnitude by injecting noise into the decoder and lost (+5.42 %, P_fine02).
        self.fine_layerscale = float(fine_layerscale)
        assert self.fine_layerscale >= 0.0, "fine_layerscale must be >= 0"
        if self.fine_layerscale > 0:
            assert float(fine_skip_init) == 0.0, (
                "--fine-layerscale and --fine-skip-init are two different fixes for the same "
                "symptom and both set the fine-skip conv init; refusing the combination")
        self.fine_to_dec = nn.Conv2d(C, up_ch[0], 1)
        nn.init.zeros_(self.fine_to_dec.bias)
        if self.fine_layerscale > 0:
            pass                                       # keep nn.Conv2d's own init; gamma does the scaling
        elif float(fine_skip_init) > 0:
            nn.init.normal_(self.fine_to_dec.weight, std=float(fine_skip_init))
        else:
            nn.init.zeros_(self.fine_to_dec.weight)
        if self.fine_layerscale > 0:                   # torch.full draws no RNG: later inits unchanged
            self.fine_ls = nn.Parameter(torch.full((up_ch[0],), self.fine_layerscale))
        self.register_buffer("fine_dir6_buf", _dir_pe(2 * lh, 2 * lw, torch.device("cpu")), persistent=False)
        self.head = _CircConv2d(up_ch[-1], 1, 3) if erp_circular else nn.Conv2d(up_ch[-1], 1, 3, padding=1)
        # fusion
        self.intra = nn.ModuleList([CondSelfAttn(C) for _ in range(rounds)])
        self.inter = nn.ModuleList([InterMicAttn(C) for _ in range(rounds)])
        self.ray_mic = RayMicAttn(C, use_bias=not no_geo_bias, qk_norm=qk_norm,
                                  ray_gate=ray_gate, view_fusion=view_fusion,
                                  logit_scale_init=logit_scale_init, geo_scale_init=geo_scale_init,
                                  ray_gate_mode=ray_gate_mode, ray_gate_init=ray_gate_init,
                                  registers=registers, scene_registers=scene_registers,
                                  geo_basis=geo_basis, geo_value_film=geo_value_film,
                                  film_lse=film_lse, geo_ray_gain=geo_ray_gain,
                                  value_moments=value_moments,
                                  tof_value_code=tof_value_code, tof_tau_inject=tof_tau_inject,
                                  geo_key_coupling=geo_key_coupling, tok_lh=lh, tok_lw=lw)
        # ---- pose re-injection (--pose-reinject). The reference adds `pose_emb(pose_feat)` to the
        # coarse tokens ONCE, in `_encode`. After that the pose reaches the fusion stack only
        # through `CondSelfAttn`'s AdaLN, which conditions each observation on ITSELF;
        # `InterMicAttn`, which attends the SAME token position across the N observations and is
        # therefore the one place where the observations are compared against each other, has no
        # pose conditioning at all. One per-channel gamma per round on the shared learnable pose
        # code: `x + gamma * pose_code`, gamma = 0 adds exactly 0.0, so the switch is bit-identical
        # at init, and dL/dgamma = <grad, pose_code> is non-zero, so it learns from step 0.
        #
        # Two alternatives rejected on measurement, both recorded on purpose:
        #   * the fine ray-lift query is pose-blind as well, and a gamma there is bit-identical at
        #     init but has a gradient of exactly 0.0 at step 0 at N = 2/4/8, because `fine_to_dec`
        #     is zero-initialised and passes W^T g = 0 back into the whole fine branch. That is the
        #     cold start `fine_layerscale` repairs, so the site is out of scope until it is combined
        #     with that switch.
        #   * an INDEPENDENT `nn.Linear(3, C)` per site would let each round learn its own pose code
        #     rather than a channel-wise rescaling of the shared one, which is strictly more
        #     expressive -- but at (3*256 + 256) * 2 = 2,048 parameters against 512 here it breaks
        #     the standing rule to re-examine anything past a few hundred. The limitation is real:
        #     the re-injected direction is tied to the single shared `pose_emb` output, and only its
        #     per-channel scale is free.
        self.pose_reinject = bool(pose_reinject)
        if self.pose_reinject:
            # torch.zeros draws no RNG, so every later init is unchanged
            self.pose_rein = nn.ParameterList([nn.Parameter(torch.zeros(C)) for _ in range(rounds)])
        # ---- content-conditioned second read (--cross-reread, 2026-10-17). The reference query is
        # `self.q + dir_mlp(dir6)`: a per-ray PRIOR that has never seen a spectrogram. It reads the
        # observation tokens once, and the four `erp` blocks that follow mix rays with each other
        # and never touch the audio again. So which tokens a ray reads is decided before the ray
        # knows anything about what is in them.
        #
        # This re-runs the SAME `ray_mic` -- same weights, no new layer -- with the refined rays as
        # the query, and mixes the result in through one zero-init per-channel gamma:
        # `h + gamma * (h2 - h)`. +C parameters. Weight sharing is the point, not a saving: the map
        # is already trained by the first read, so unlike the fine branch there is no cold start
        # (`fine_to_dec`'s zero init is what made a gamma there dead at step 0; here `h2 - h` is a
        # real non-zero tensor on the first step, so dL/dgamma = <g, h2-h> is alive).
        #
        # Known limitation, recorded: if the shared query map turns out to be wrong for a refined
        # query, the repair is a Linear(C, C) query adapter at +65.8 k, which would be
        # CAPACITY-CONFOUNDED against a 15.35 M control. This cell deliberately buys the cheap,
        # unconfounded version first. Second note: `ray_mic.collect_stats` is written twice per
        # forward when this is on, and the SECOND read wins -- any diagnostic that reads those
        # stats is reading the re-read, not the first read.
        self.cross_reread = bool(cross_reread)
        if self.cross_reread:
            self.reread_ls = nn.Parameter(torch.zeros(C))

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
        skip = self.fine_to_dec(fine_erp)
        if self.fine_layerscale > 0:
            skip = skip * self.fine_ls.view(1, -1, 1, 1)                  # LayerScale (CaiT)
        x = self.up_stages[0](x) + skip
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
        # (nv,C) pose code for --pose-reinject, shared by every round. `pose_emb` is already
        # evaluated twice in the reference forward, so this is a third evaluation only when the
        # switch is on, and it draws no RNG.
        pose_code = self.pose_emb(pose_feat) if self.pose_reinject else None
        for r, (intra, inter) in enumerate(zip(self.intra, self.inter)):  # alternating attention
            B_, N, M, C = F4.shape
            F4 = intra(F4.reshape(B_ * N, M, C), cond).reshape(B_, N, M, C)
            if pose_code is not None:
                # gamma = 0 at init -> the added term is exactly 0.0 -> bitwise no-op
                F4 = F4 + (self.pose_rein[r] * pose_code).view(1, N, 1, C)
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
        if self.cross_reread:
            # gamma = 0 at init -> `h + 0.0 * (h2 - h)` adds exactly 0.0 -> bitwise no-op
            h2 = self.ray_mic(h, F4.reshape(B, self.nv * self.M, self.C), dir3, poses,
                              self.M, obs_valid)
            h = h + self.reread_ls * (h2 - h)
        return self._decode(h, fine, poses, obs_valid)
