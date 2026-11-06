"""Pair (same-heading, two-ear) input features for OAA -- `--pair-feat` (2026-10-27).

The reference model encodes every (yaw, ear) observation on its own and first compares the two
ears of one heading at the 16x32 token grid. These features hand the encoder the inter-ear
relation at FULL time-frequency resolution, as a second input plane per observation:

  none      reference: spec returned untouched (the object itself, so the path is bit-identical)
  ild       plane 2 = pair level contrast c = (|own| - |partner|) / (|own| + |partner| + EPS)
            in [-1, 1], antisymmetric within the pair (c_L = -c_R), a monotone function of the
            ILD in dB: c = tanh(ln(|own|/|partner|) / 2) = tanh(ILD_dB * ln(10) / 40). Bounded and
            EPS-insensitive where a log-ratio would blow up on silent bins.
  ildmin    plane 2 = UNSIGNED pair level ratio min(|own|, |partner|) / (max(|own|, |partner|) + EPS)
            in [0, 1], SYMMETRIC within the pair (both ears receive the SAME plane). The sign-free
            counterpart of `ild`: min/max = exp(-|ln(|own|/|partner|)|) = (1 - |c|) / (1 + |c|), so
            exactly one thing changes against `ild` -- the sign (2026-11-03). Motivation: the
            direct-sound ILD is a constant +-18.45 dB per heading on Gibson, Replica AND MP3D
            (heading-signature probe 2026-11-03), so the sign of the `ild` plane is a free pose /
            ear label; `ildmin` keeps the magnitude of the level difference and drops that label.
            NOT the raw min(|L|, |R|): that is a second magnitude spectrogram and nearly duplicates
            plane 1; only the ratio carries new information.
  partner   plane 2 = the partner ear's raw magnitude (pair stacking: the access a channel-stacked
            CNN has, without the ILD inductive bias)
  ipd       planes 2,3 = interaural PHASE, E135 definition (e135/data_0422_bin.py cue_channels):
            cos(dphi)*SCALE and sgn_ear*sin(dphi)*SCALE with dphi = angle(S_L * conj(S_R)) on the raw
            complex STFT, sgn = +1 for ear 0 (L) / -1 for ear 1 (R). Phase is not in the magnitude
            spec, so the loader computes the two planes (data module EMIT_IPD) and build() receives
            them as `ipd` (B, N, 2, H, W), already ear-signed and in `chans` order (2026-11-01).
  ildipd    planes 2,3,4 = [ILD contrast c (as `ild`), cos IPD, sgn_ear*sin IPD (as `ipd`)]: level
            AND phase, the E135 `bin4` cue set (2026-11-02). The ILD plane is this module's bounded
            contrast c, NOT E135's scaled log10 ratio; the two phase planes are the loader's `ipd`
            planes unchanged, so `ildipd` planes 3,4 == `ipd` planes 2,3 bit for bit.
  earmean   NO second plane; both ears of a pair are replaced by their mean (L+R)/2. An input
            ablation that removes every inter-ear difference while keeping the monaural content
            and the channel count (= the `ear_mean` perturbation of the 2026-10-27 diagnostic,
            applied at training time).

Channel layout: the data module's `_CH[mode]` list of (yaw-slot, ear) in spec channel order. A
channel without a partner (single-ear modes `al`/`ar`) gets c = 0 / partner = 0 / unchanged mean.
Output layout for in_ch 2: channel 2i = magnitude of observation i, channel 2i+1 = its pair plane,
which is exactly the `spec.view(B, nv, in_ch, H, W)` contract of `OAAv2Depth._encode`.

`dropped` (list of observation indices zeroed by --subset-aug): both planes of a dropped
observation are zeroed AND the partner's pair plane is zeroed, so a dropped ear cannot leak back
in through its partner's contrast. The planes are always built from the CLEAN spec first.
"""
import torch

KINDS = ("none", "ild", "partner", "earmean", "stereo", "stereoild", "ipd", "ildipd", "ildmin",
         "itd", "stackall")
EPS = 1e-6
ITD_MAX = 16.0          # samples; lag half-width of the `itd` GCC-PHAT search


def in_ch(kind, nviews=None):
    """Input planes per observation the model must be built with.

    `stackall` needs nviews: every loader channel becomes one plane of the single observation.
    """
    assert kind in KINDS, f"pair_feat {kind} not in {KINDS}"
    if kind == "stackall":
        assert nviews, "pair_feat stackall needs nviews to size the stacked observation"
        return int(nviews)
    if kind in ("ild", "partner", "stereo", "ildmin", "itd"):
        return 2
    if kind in ("stereoild", "ipd"):
        return 3
    if kind == "ildipd":
        return 4
    return 1


def folds_views(kind):
    """True when the model sees FEWER observations than the loader hands over."""
    return kind in ("stereo", "stereoild", "stackall")


def folds_pairs(kind):
    """True for the (L,R) PAIR fold only -- the modes whose observation unit is two channels.
    `stackall` is excluded on purpose: there one CHANNEL is still one observation (one plane),
    so --subset-aug keeps counting and dropping in channel units, exactly like ctrl."""
    return kind in ("stereo", "stereoild")


def fold_to(kind, nviews):
    """Observation count the model must be built with for `nviews` loader channels."""
    if kind == "stackall":
        return 1
    if folds_pairs(kind):
        assert nviews % 2 == 0, f"{kind} needs an even nviews, got {nviews}"
        return nviews // 2
    return nviews


def needs_ipd(kind):
    """True when build() needs the loader's phase planes (data module EMIT_IPD must be switched on)."""
    return kind in ("ipd", "ildipd", "itd")


def partner_index(chans):
    """chans: list of (slot, ear) -> list of partner index per channel (-1 = no partner)."""
    slots = {}
    for i, (s, e) in enumerate(chans):
        slots.setdefault(int(s), {})[int(e)] = i
    return [slots[int(s)].get(1 - int(e), -1) for s, e in chans]


def build(spec, chans, kind, dropped=None, ipd=None):
    """spec (B, N, H, W) magnitude spectrograms in `chans` order -> model input.
    none -> spec itself; earmean -> (B, N, H, W); ild / partner / ildmin -> (B, 2N, H, W);
    ipd -> (B, 3N, H, W) from the loader's `ipd` planes (B, N, 2, H, W);
    ildipd -> (B, 4N, H, W) = per observation [mag, ILD contrast, cos IPD, sgn*sin IPD]."""
    assert kind in KINDS, f"pair_feat {kind} not in {KINDS}"
    if kind == "none":
        return spec
    B, N, H, W = spec.shape
    assert N == len(chans), f"spec has {N} channels but chans lists {len(chans)}"
    pidx = partner_index(chans)
    has = torch.tensor([j >= 0 for j in pidx], device=spec.device).view(1, N, 1, 1)
    gather = torch.tensor([j if j >= 0 else i for i, j in enumerate(pidx)], device=spec.device)
    part = spec.index_select(1, gather)                                   # (B, N, H, W) partner (or self)
    if kind == "earmean":
        out = torch.where(has, 0.5 * (spec + part), spec)
        if dropped:
            out = out.clone()
            out[:, list(dropped)] = 0
        return out
    if kind == "stackall":
        # Architecture ablation, Changwoo 2026-11-06. Hand the encoder ALL channels as the planes
        # of ONE observation instead of one posed observation per channel. The TENSOR IS UNCHANGED:
        # the model is built with nviews=1 and in_ch=N, so `spec.view(B, 1, N, H, W)` reads exactly
        # the same bytes as a single stacked token. This therefore ablates the TOKENISATION only --
        # the per-view pose/ray embedding and the cross-view attention -- and nothing about the
        # input content. At 8 channels the four headings collapse into one token and the surviving
        # pose is the first channel's (0 deg, L): unlike the 2-channel stereo fold a pose IS being
        # invented here, which is the thing under test.
        out = spec
        if dropped:
            out = out.clone()
            out[:, list(dropped)] = 0
        return out
    if kind in ("stereo", "stereoild"):
        # Fold each (slot, L/R) pair into a single observation whose planes are [L, R(, ILD)].
        # 2026-10-31 (Changwoo): the 2-channel rig has both ears at the SAME heading, so one pose
        # describes the folded observation and no pose has to be invented. Mirrored duplication of
        # the existing per-ear layout ([L,R] and [R,L]) disappears, and the view-fusion stage sees
        # one token instead of two. Pairs are taken in ascending slot order, ear 0 (L) first.
        slots = {}
        for i, (s, e) in enumerate(chans):
            slots.setdefault(int(s), {})[int(e)] = i
        order = sorted(slots)
        assert all(len(slots[s]) == 2 for s in order), \
            f"stereo needs both ears per slot; got {[(s, sorted(slots[s])) for s in order]}"
        li = torch.tensor([slots[s][0] for s in order], device=spec.device)
        ri = torch.tensor([slots[s][1] for s in order], device=spec.device)
        L = spec.index_select(1, li)
        Rr = spec.index_select(1, ri)
        planes = [L, Rr]
        if kind == "stereoild":
            planes.append((L - Rr) / (L + Rr + EPS))
        k = len(planes)
        out = torch.stack(planes, 2).reshape(B, k * len(order), H, W)
        if dropped:
            out = out.clone()
            drop_slots = {int(chans[i][0]) for i in dropped}
            for n, s in enumerate(order):
                if s in drop_slots:
                    out[:, k * n: k * (n + 1)] = 0
        return out
    if kind == "itd":
        # One broadband delay per observation instead of the full per-bin phase (2026-11-06).
        # `ipd` gives the model phase at every frequency bin; `itd` collapses that to a SINGLE
        # GCC-PHAT delay and broadcasts it as a constant plane, so the only thing that changes
        # against `ipd` is how much of the phase survives. Measured earlier on Gibson: the best
        # single delay explains 0.42 of the phase, so this arm is the lossy projection on purpose.
        #
        # The loader hands us (cos phi, sgn_ear * sin phi) per bin. GCC-PHAT over a fixed lag grid:
        #   R(tau) = sum_h cos(phi_h + omega_h * tau),  omega_h = pi * h / (H - 1)
        # with the unit-magnitude cross spectrum averaged over time frames first (stationary delay).
        # Lag grid is +-ITD_MAX samples at quarter-sample steps; ITD_MAX 16 covers the measured
        # +-7-sample interaural range with margin. Output plane = tau_hat / ITD_MAX in [-1, 1].
        assert ipd is not None, "pair_feat itd needs the loader's `ipd` planes (data module EMIT_IPD)"
        assert tuple(ipd.shape) == (B, N, 2, H, W), f"ipd planes {tuple(ipd.shape)} vs spec {tuple(spec.shape)}"
        cos_p = ipd[:, :, 0].mean(-1)                                     # (B, N, H) time-averaged
        sin_p = ipd[:, :, 1].mean(-1)
        omega = torch.linspace(0.0, torch.pi, H, device=spec.device, dtype=spec.dtype)
        lags = torch.arange(-4 * ITD_MAX, 4 * ITD_MAX + 1, device=spec.device,
                            dtype=spec.dtype) * 0.25                      # (L,) quarter samples
        wt = omega.unsqueeze(1) * lags.unsqueeze(0)                       # (H, L)
        # sum_h cos(phi_h + omega_h tau) = sum_h [cos phi_h cos(wt) - sin phi_h sin(wt)]
        r = cos_p @ torch.cos(wt) - sin_p @ torch.sin(wt)                 # (B, N, L)
        tau = lags[r.argmax(-1)] / ITD_MAX                                # (B, N) in [-1, 1]
        c = tau.view(B, N, 1, 1).expand(B, N, H, W)
        c = torch.where(has, c, torch.zeros_like(c))
        out = torch.stack([spec, c], 2).reshape(B, 2 * N, H, W)
        if dropped:
            out = out.clone()
            for i in dropped:
                out[:, 2 * i: 2 * i + 2] = 0
                j = pidx[i]
                if j >= 0:
                    out[:, 2 * j + 1] = 0
        return out
    if kind == "ipd":
        # A channel without a partner gets zero planes, exactly like ild; a dropped observation
        # loses all three planes and its partner's two phase planes (same rule as ild's plane 2).
        assert ipd is not None, "pair_feat ipd needs the loader's `ipd` planes (data module EMIT_IPD)"
        assert tuple(ipd.shape) == (B, N, 2, H, W), f"ipd planes {tuple(ipd.shape)} vs spec {tuple(spec.shape)}"
        c = torch.where(has.unsqueeze(2), ipd, torch.zeros_like(ipd))
        out = torch.cat([spec.unsqueeze(2), c], 2).reshape(B, 3 * N, H, W)
        if dropped:
            out = out.clone()
            for i in dropped:
                out[:, 3 * i: 3 * i + 3] = 0
                j = pidx[i]
                if j >= 0:
                    out[:, 3 * j + 1: 3 * j + 3] = 0
        return out
    if kind == "ildipd":
        # E135 bin4 (2026-11-02): [mag, ILD contrast, cos IPD, sgn*sin IPD]. The ILD plane is the
        # `ild` branch's c verbatim and the phase planes are the `ipd` branch's verbatim; a channel
        # without a partner gets three zero planes; a dropped observation loses all four planes and
        # its partner loses its three pair planes (planes 1..3), the same rule as ild / ipd.
        assert ipd is not None, "pair_feat ildipd needs the loader's `ipd` planes (data module EMIT_IPD)"
        assert tuple(ipd.shape) == (B, N, 2, H, W), f"ipd planes {tuple(ipd.shape)} vs spec {tuple(spec.shape)}"
        c = (spec - part) / (spec + part + EPS)
        c = torch.where(has, c, torch.zeros_like(c))
        ph = torch.where(has.unsqueeze(2), ipd, torch.zeros_like(ipd))
        out = torch.cat([spec.unsqueeze(2), c.unsqueeze(2), ph], 2).reshape(B, 4 * N, H, W)
        if dropped:
            out = out.clone()
            for i in dropped:
                out[:, 4 * i: 4 * i + 4] = 0
                j = pidx[i]
                if j >= 0:
                    out[:, 4 * j + 1: 4 * j + 4] = 0
        return out
    if kind == "ild":
        c = (spec - part) / (spec + part + EPS)
    elif kind == "ildmin":
        # unsigned level ratio in [0, 1], identical for both ears of a pair (2026-11-03); everything
        # below (no-partner zeroing, stacking, `dropped`) is the `ild` path unchanged
        c = torch.minimum(spec, part) / (torch.maximum(spec, part) + EPS)
    else:                                                                 # partner
        c = part
    c = torch.where(has, c, torch.zeros_like(c))
    out = torch.stack([spec, c], 2).reshape(B, 2 * N, H, W)
    if dropped:
        out = out.clone()
        for i in dropped:
            out[:, 2 * i: 2 * i + 2] = 0
            j = pidx[i]
            if j >= 0:
                out[:, 2 * j + 1] = 0
    return out
