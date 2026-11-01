"""Pair (same-heading, two-ear) input features for OAA -- `--pair-feat` (2026-10-27).

The reference model encodes every (yaw, ear) observation on its own and first compares the two
ears of one heading at the 16x32 token grid. These features hand the encoder the inter-ear
relation at FULL time-frequency resolution, as a second input plane per observation:

  none      reference: spec returned untouched (the object itself, so the path is bit-identical)
  ild       plane 2 = pair level contrast c = (|own| - |partner|) / (|own| + |partner| + EPS)
            in [-1, 1], antisymmetric within the pair (c_L = -c_R), a monotone function of the
            ILD in dB: c = tanh(ln(|own|/|partner|) / 2) = tanh(ILD_dB * ln(10) / 40). Bounded and
            EPS-insensitive where a log-ratio would blow up on silent bins.
  partner   plane 2 = the partner ear's raw magnitude (pair stacking: the access a channel-stacked
            CNN has, without the ILD inductive bias)
  ipd       planes 2,3 = interaural PHASE, E135 definition (e135/data_0422_bin.py cue_channels):
            cos(dphi)*SCALE and sgn_ear*sin(dphi)*SCALE with dphi = angle(S_L * conj(S_R)) on the raw
            complex STFT, sgn = +1 for ear 0 (L) / -1 for ear 1 (R). Phase is not in the magnitude
            spec, so the loader computes the two planes (data module EMIT_IPD) and build() receives
            them as `ipd` (B, N, 2, H, W), already ear-signed and in `chans` order (2026-11-01).
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

KINDS = ("none", "ild", "partner", "earmean", "stereo", "stereoild", "ipd")
EPS = 1e-6


def in_ch(kind):
    """Input planes per observation the model must be built with."""
    assert kind in KINDS, f"pair_feat {kind} not in {KINDS}"
    if kind in ("ild", "partner", "stereo"):
        return 2
    if kind in ("stereoild", "ipd"):
        return 3
    return 1


def folds_views(kind):
    """True when build() collapses each (L,R) pair into ONE observation (nviews halves)."""
    return kind in ("stereo", "stereoild")


def partner_index(chans):
    """chans: list of (slot, ear) -> list of partner index per channel (-1 = no partner)."""
    slots = {}
    for i, (s, e) in enumerate(chans):
        slots.setdefault(int(s), {})[int(e)] = i
    return [slots[int(s)].get(1 - int(e), -1) for s, e in chans]


def build(spec, chans, kind, dropped=None, ipd=None):
    """spec (B, N, H, W) magnitude spectrograms in `chans` order -> model input.
    none -> spec itself; earmean -> (B, N, H, W); ild / partner -> (B, 2N, H, W);
    ipd -> (B, 3N, H, W) from the loader's `ipd` planes (B, N, 2, H, W)."""
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
    if kind == "ild":
        c = (spec - part) / (spec + part + EPS)
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
