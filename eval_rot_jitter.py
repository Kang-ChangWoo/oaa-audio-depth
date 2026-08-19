"""Per-mic random-rotation (rig-geometry jitter) eval on replica_0422_rot30_test.

Front binaural pair stays on the 90-degree grid (theta in {0,90,180,270}, GT at theta) so the
front view is in-distribution; mics 1-3 are individually jittered off their nominal +90/180/270
slots by fixed random patterns (deterministic, listed below). L/R of a mic always move together.

OAA is evaluated twice per batch: told the TRUE per-mic poses vs the nominal (blind) poses —
isolates whether pose conditioning actually exploits rig geometry. eco is inherently blind.

  DATA_MODULE=data_0422 CUDA_VISIBLE_DEVICES=5 python3 eval_rot_jitter.py \
      --model oaa --ckpt comparison/oaa_r8_fin/best.pth
"""
import os, math, json, argparse
import torch
from torch.utils.data import Dataset, DataLoader

os.environ.setdefault("DATA_MODULE", "data_0422")
import data_0422 as dm

ROOT = os.environ.get("REPLICA_ROT30_ROOT", "data/replica_0422_rot30_test")   # rotated-heading test set
SCENES = ["apartment_2", "frl_apartment_5", "office_4"]
THETAS = (0, 90, 180, 270)
PATTERNS = [(30, -30, 45), (-45, 30, -30), (45, -60, 30),
            (-30, 45, 60), (60, -45, -60), (-60, 60, -45)]   # (d1,d2,d3) for mic slots +90/+180/+270
# front-jitter variant: (d0,d1,d2,d3) — ALL four mics off-grid, GT stays at the grid theta,
# so no mic looks exactly along the target frame (true-pose slot0 yaw != 0 is pure extrapolation)
PATTERNS4 = [(-30, 30, -30, 45), (45, -45, 30, -30), (-45, 45, -60, 30),
             (60, -30, 45, 60), (-60, 60, -45, -60), (30, -60, 60, -45)]
KEYS = ["MAE", "MAE_plain", "RMSE", "AbsRel", "log10", "delta1", "delta2", "delta3"]
BANDS = [("near<3", 0, 3), ("mid3-6", 3, 6), ("far>6", 6, 10)]


def cos_lat(h, device):
    v = torch.arange(h, device=device, dtype=torch.float32)
    return torch.cos((math.pi / 2) - (v + 0.5) / h * math.pi).clamp(min=1e-3)


class JitterSet(Dataset):
    def __init__(self, deltas, wave_ch=0):
        self.samples = [(sc, loc, th) for sc in SCENES for loc in range(100) for th in THETAS]
        d0, rest = (deltas[0], deltas[1:]) if len(deltas) == 4 else (0, deltas)
        self.offsets = [d0] + [90 * (k + 1) + d for k, d in enumerate(rest)]    # slot yaw offsets
        self.wave_ch = wave_ch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sc, loc, th = self.samples[i]
        wav = [dm._load_wave1(f"{ROOT}/{sc}/audio_wav/audio_{loc:03d}_{(th + o) % 360:03d}.wav")
               for o in self.offsets]
        spec = torch.cat([dm._stft_mag(w) for w in wav], 0)                     # (8,256,512)
        d, mask = dm._load_depth(f"{ROOT}/{sc}/erp_depth/erp_depth_{loc:03d}_{th:03d}.npy")
        out = {"spec": spec, "depth": d, "mask": mask}
        if self.wave_ch:
            out["wave"] = torch.cat(wav, 0)[: self.wave_ch]
        return out


def poses_of(offsets):
    return [(math.radians(o % 360), s) for o in offsets for s in (-1.0, 1.0)]


class Meter:
    def __init__(self):
        self.acc = {k: 0.0 for k in KEYS}; self.n = 0
        self.be = {b[0]: [0.0, 0.0] for b in BANDS}

    def add(self, D, gt, mask, wlat):
        w = wlat * mask; B = D.shape[0]
        pi = lambda num, den: (num.flatten(1).sum(1) / den.flatten(1).sum(1).clamp(min=1e-6))
        err = (D - gt).abs()
        self.acc["MAE"] += float(pi(err * w, w).mean()) * B
        self.acc["MAE_plain"] += float(pi(err * mask, mask).mean()) * B
        self.acc["RMSE"] += float(pi(err ** 2 * w, w).sqrt().mean()) * B
        self.acc["AbsRel"] += float(pi(err / gt.clamp(min=1e-3) * w, w).mean()) * B
        self.acc["log10"] += float(pi((torch.log10(D.clamp(min=1e-3)) - torch.log10(gt.clamp(min=1e-3))).abs() * w, w).mean()) * B
        rt = torch.maximum(D / gt.clamp(min=1e-3), gt / D.clamp(min=1e-3))
        for j, k in enumerate(["delta1", "delta2", "delta3"]):
            self.acc[k] += float(pi((rt < 1.25 ** (j + 1)).float() * w, w).mean()) * B
        for nm, lo, hi in BANDS:
            bm = w * (gt >= lo) * (gt < hi)
            self.be[nm][0] += float((err * bm).sum()); self.be[nm][1] += float(bm.sum())
        self.n += B

    def result(self):
        r = {k: v / self.n for k, v in self.acc.items()}
        r.update({nm: s / max(c, 1e-6) for nm, (s, c) in self.be.items()})
        r["n"] = self.n
        return r


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["oaa", "eco", "bat"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--front-jitter", action="store_true", help="jitter the front mic too (PATTERNS4)")
    ap.add_argument("--tag", default="", help="suffix for the output json (e.g. _wnone)")
    a = ap.parse_args()
    pats = PATTERNS4 if a.front_jitter else PATTERNS
    device = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = ck["args"]; md = args.get("max_depth", 10.0)
    if a.model in ("oaa", "bat"):
        import eval as ev
        net, mode, *_ = ev.build(args)
        net.load_state_dict(ck["state_dict"]); net.to(device).eval()
        wave_ch = 0
    else:
        from model.echodiffusion import EchoDiffusionDepth
        mode = args.get("mode", "r2"); wm = args.get("wave_mode", "std")
        wave_ch = dm.IN_CH[mode] if wm == "all" else 2   # "none" also passes a dummy 2ch (ignored by the model)
        net = EchoDiffusionDepth(in_ch=dm.IN_CH[mode], wave_mode="none" if wm == "none" else "cide",
                                 wave_ch=wave_ch, faithful=(args.get("port", "enhanced") == "faithful"))
        net.load_state_dict(ck["state_dict"]); net.to(device).eval()
    assert mode == "r8", f"jitter eval is r8-only (got {mode})"
    nominal = poses_of([0, 90, 180, 270])
    print(f"[cfg] model={a.model} ckpt={a.ckpt} front_jitter={a.front_jitter} patterns={pats}", flush=True)

    out = {}
    variants = ["true", "blind"] if a.model == "oaa" else ["blind"]
    meters_all = {v: Meter() for v in variants}
    for deltas in pats:
        ds = JitterSet(deltas, wave_ch)
        dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)
        true_poses = poses_of(ds.offsets)
        wlat = cos_lat(dm.H, device).view(1, 1, dm.H, 1)
        meters = {v: Meter() for v in variants}
        for b in dl:
            spec = b["spec"].to(device)
            gt = b["depth"].to(device) * dm.MAX_DEPTH; mask = b["mask"].to(device)
            if a.model == "oaa":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    Dt = net(spec, view_poses=true_poses).float() * md
                    Db = net(spec, view_poses=nominal).float() * md
                meters["true"].add(Dt, gt, mask, wlat); meters_all["true"].add(Dt, gt, mask, wlat)
                meters["blind"].add(Db, gt, mask, wlat); meters_all["blind"].add(Db, gt, mask, wlat)
            elif a.model == "bat":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    D = net(spec).float() * md
                meters["blind"].add(D, gt, mask, wlat); meters_all["blind"].add(D, gt, mask, wlat)
            else:
                D = net(spec, b["wave"].to(device)).float() * md
                meters["blind"].add(D, gt, mask, wlat); meters_all["blind"].add(D, gt, mask, wlat)
        key = "j" + "_".join(str(d) for d in deltas)
        out[key] = {v: meters[v].result() for v in variants}
        print(f"[{key}] " + " | ".join(f"{v}: MAE={out[key][v]['MAE']:.4f} d1={out[key][v]['delta1']:.4f}"
                                       for v in variants), flush=True)
    out["ALL"] = {v: meters_all[v].result() for v in variants}
    print("[ALL] " + " | ".join(f"{v}: MAE={out['ALL'][v]['MAE']:.4f} d1={out['ALL'][v]['delta1']:.4f}"
                                for v in variants), flush=True)
    jp = os.path.join("comparison", f"rotjitter_eval_{a.model}_r8{a.tag}{'_front' if a.front_jitter else ''}.json")
    json.dump(out, open(jp, "w"), indent=2)
    print(f"[done] -> {jp}", flush=True)


if __name__ == "__main__":
    main()
