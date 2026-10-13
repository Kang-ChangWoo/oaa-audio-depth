"""Depth metrics: cos-latitude-weighted, PER-IMAGE (batch-invariant) MAE / RMSE / AbsRel / log10 / delta1-3,
plus near(<3m)/mid(3-6m)/far(>6m) band MAE. Depths in metres; `mask` = valid pixels."""
import math
import torch

KEYS = ["MAE", "MAE_plain", "RMSE", "AbsRel", "log10", "delta1", "delta2", "delta3"]
BANDS = [("near<3", 0, 3), ("mid3-6", 3, 6), ("far>6", 6, 10)]


def cos_lat(h, device):
    """Per-row cos(latitude) weights for an ERP map of height h (clamped away from 0 at the poles)."""
    v = torch.arange(h, device=device, dtype=torch.float32)
    return torch.cos((math.pi / 2) - (v + 0.5) / h * math.pi).clamp(min=1e-3)


def depth_loss(D, gt, mask, kind, wlat):
    """Training objective on NORMALISED depth (metres only for reporting: multiply by max_depth).

    `plain`     the reference masked L1, pooled over every valid pixel in the batch at once.
    `spherical` cos-latitude-weighted L1 normalised PER IMAGE first, then averaged over the batch --
                the same aggregation the reported MAE uses (`MetricAccumulator`), unlike `plain`
                and unlike `train_oaa.quick_val`, which both pool globally.

    Returns None when not one sample in the batch has a single weighted-valid pixel, so the caller
    can drop the micro-batch instead of stepping the optimizer on a NaN.
    """
    if kind == "plain":
        return ((D.float() - gt).abs() * mask).sum() / mask.sum().clamp(min=1e-6)
    w = mask.float() * wlat
    num = ((D.float() - gt.float()).abs() * w).flatten(1).sum(1)
    den = w.flatten(1).sum(1)
    valid = den > 0
    if not bool(valid.any()):
        return None
    return (num[valid] / den[valid].clamp_min(1e-6)).mean()


class MetricAccumulator:
    """acc = MetricAccumulator(device); acc.update(D, gt, mask) per batch; acc.result() -> dict (KEYS + bands)."""

    def __init__(self, device, h=256):
        self.wlat = cos_lat(h, device).view(1, 1, h, 1)
        self.acc = {k: 0.0 for k in KEYS}; self.n = 0
        self.be = {b[0]: [0.0, 0.0] for b in BANDS}

    @torch.no_grad()
    def update(self, D, gt, mask):
        """D, gt: (B,1,H,W) metres; mask: (B,1,H,W) valid pixels (0/1)."""
        w = self.wlat * mask; B = D.shape[0]; acc = self.acc
        pi = lambda num, den: (num.flatten(1).sum(1) / den.flatten(1).sum(1).clamp(min=1e-6))
        acc["MAE"] += float(pi((D - gt).abs() * w, w).mean()) * B
        acc["MAE_plain"] += float(pi((D - gt).abs() * mask, mask).mean()) * B   # unweighted (mask only)
        acc["RMSE"] += float(pi(((D - gt) ** 2) * w, w).clamp(min=0).sqrt().mean()) * B
        acc["AbsRel"] += float(pi((D - gt).abs() / gt.clamp(min=0.1) * w, w).mean()) * B
        acc["log10"] += float(pi((torch.log10(D.clamp(min=0.1)) - torch.log10(gt.clamp(min=0.1))).abs() * w, w).mean()) * B
        rt = torch.maximum(D.clamp(min=0.1) / gt.clamp(min=0.1), gt.clamp(min=0.1) / D.clamp(min=0.1))
        acc["delta1"] += float(pi((rt < 1.25).float() * w, w).mean()) * B
        acc["delta2"] += float(pi((rt < 1.25 ** 2).float() * w, w).mean()) * B
        acc["delta3"] += float(pi((rt < 1.25 ** 3).float() * w, w).mean()) * B
        self.n += B
        err = (D - gt).abs()
        for nm, lo, hi in BANDS:
            bm = mask * (gt >= lo) * (gt < hi)
            self.be[nm][0] += (err * bm).sum().item(); self.be[nm][1] += bm.sum().item()

    def result(self):
        out = {k: self.acc[k] / max(self.n, 1) for k in KEYS}
        for nm in self.be:
            out[nm] = self.be[nm][0] / max(self.be[nm][1], 1e-6)
        return out
