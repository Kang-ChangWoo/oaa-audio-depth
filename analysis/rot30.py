"""Rig-rotation robustness eval on replica_0422_rot30_test (fin r8 models).

The test set (3 test scenes x 100 locations x 16 yaws) provides fronts at 30-degree and
45-degree granularity; the 8ch rig is always the front's 4-yaw group (+0/90/180/270, [L,R]
each), so this measures how the fin models behave when the whole rig is rotated off the
90-degree-multiple fronts seen in training. Spec/depth/metric recipe identical to
data_0422 + eval.py (cos-lat weighted, per-image).

  DATA_MODULE=data_0422 CUDA_VISIBLE_DEVICES=5 python analysis/rot30.py \
      --model oaa --ckpt comparison/oaa_r8_fin/best.pth
  DATA_MODULE=data_0422 CUDA_VISIBLE_DEVICES=6 HF_HOME=... echodiff_env/bin/python analysis/rot30.py \
      --model eco --ckpt comparison/eco_r8_fin/best.pth
"""
# --- repo-root bootstrap: importable root modules (eval, data_*, model) + relative comparison/ paths
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
_os.chdir(ROOT)
import os, math, json, argparse
import torch
from torch.utils.data import Dataset, DataLoader

os.environ.setdefault("DATA_MODULE", "data_0422")
import data_0422 as dm

ROOT = os.environ.get("REPLICA_ROT30_ROOT", "data/replica_0422_rot30_test")   # rotated-heading test set
SCENES = ["apartment_2", "frl_apartment_5", "office_4"]
FAMILIES = {"rot90": range(0, 360, 90), "rot45": range(0, 360, 45), "rot30": range(0, 360, 30)}
KEYS = ["MAE", "MAE_plain", "RMSE", "AbsRel", "log10", "delta1", "delta2", "delta3"]
BANDS = [("near<3", 0, 3), ("mid3-6", 3, 6), ("far>6", 6, 10)]


def cos_lat(h, device):
    v = torch.arange(h, device=device, dtype=torch.float32)
    return torch.cos((math.pi / 2) - (v + 0.5) / h * math.pi).clamp(min=1e-3)


class RotSet(Dataset):
    def __init__(self, yaws, mode, wave_ch=0):
        self.samples = [(sc, loc, y) for sc in SCENES for loc in range(100) for y in yaws]
        self.chans = dm._CH[mode]                # rig slots follow the mode, e.g. r6 = +0/90/270
        self.offs = sorted({o for o, _ in self.chans})
        self.wave_ch = wave_ch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sc, loc, yaw = self.samples[i]
        wav = {o: dm._load_wave1(f"{ROOT}/{sc}/audio_wav/audio_{loc:03d}_{(yaw + 90 * o) % 360:03d}.wav")
               for o in self.offs}
        sp = {o: dm._stft_mag(wav[o]) for o in self.offs}
        d, mask = dm._load_depth(f"{ROOT}/{sc}/erp_depth/erp_depth_{loc:03d}_{yaw:03d}.npy")
        out = {"spec": torch.cat([sp[o][e:e + 1] for o, e in self.chans], 0), "depth": d, "mask": mask}
        if self.wave_ch:
            out["wave"] = torch.cat([wav[o][e:e + 1] for o, e in self.chans], 0)[: self.wave_ch]
        return out


@torch.no_grad()
def run_family(fwd, yaws, mode, wave_ch, device, md):
    dl = DataLoader(RotSet(yaws, mode, wave_ch), batch_size=16, shuffle=False, num_workers=8)
    wlat = cos_lat(dm.H, device).view(1, 1, dm.H, 1)
    acc = {k: 0.0 for k in KEYS}; n = 0
    be = {b[0]: [0.0, 0.0] for b in BANDS}
    for b in dl:
        D = fwd(b) * md
        gt = b["depth"].to(device) * dm.MAX_DEPTH
        w = wlat * b["mask"].to(device); B = D.shape[0]
        pi = lambda num, den: (num.flatten(1).sum(1) / den.flatten(1).sum(1).clamp(min=1e-6))
        err = (D - gt).abs()
        acc["MAE"] += float(pi(err * w, w).mean()) * B
        acc["MAE_plain"] += float(pi(err * b["mask"].to(device), b["mask"].to(device)).mean()) * B
        acc["RMSE"] += float(pi(err ** 2 * w, w).sqrt().mean()) * B
        acc["AbsRel"] += float(pi(err / gt.clamp(min=1e-3) * w, w).mean()) * B
        acc["log10"] += float(pi((torch.log10(D.clamp(min=1e-3)) - torch.log10(gt.clamp(min=1e-3))).abs() * w, w).mean()) * B
        rt = torch.maximum(D / gt.clamp(min=1e-3), gt / D.clamp(min=1e-3))
        acc["delta1"] += float(pi((rt < 1.25).float() * w, w).mean()) * B
        acc["delta2"] += float(pi((rt < 1.25 ** 2).float() * w, w).mean()) * B
        acc["delta3"] += float(pi((rt < 1.25 ** 3).float() * w, w).mean()) * B
        for nm, lo, hi in BANDS:
            bm = w * (gt >= lo) * (gt < hi)
            be[nm][0] += float((err * bm).sum()); be[nm][1] += float(bm.sum())
        n += B
    res = {k: v / n for k, v in acc.items()}
    res.update({nm: s / max(c, 1e-6) for nm, (s, c) in be.items()})
    res["n"] = n
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["oaa", "eco", "bat"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subsets", default="", help='override families, e.g. "off30:30,120,210,300;off60:60,150,240,330"')
    ap.add_argument("--tag", default="", help="suffix for the output json when using --subsets")
    a = ap.parse_args()
    if a.subsets:
        FAMILIES.clear()
        for part in a.subsets.split(";"):
            nm, ys = part.split(":")
            FAMILIES[nm] = [int(y) for y in ys.split(",")]
    device = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = ck["args"]; md = args.get("max_depth", 10.0)
    if a.model in ("oaa", "bat"):
        from core.data import get_data_module
        from core.ckpt import build
        _DM = get_data_module()
        net, mode, _n, kind, poses = build(args, _DM)
        net.load_state_dict(ck["state_dict"]); net.to(device).eval()
        wave_ch = 0

        def fwd(b):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = b["spec"].to(device)
                return (net(x, view_poses=poses) if poses is not None else net(x)).float()
    else:
        from model.echodiffusion import EchoDiffusionDepth
        mode = args.get("mode", "r2"); in_ch = dm.IN_CH[mode]
        wm = args.get("wave_mode", "std")
        wave_ch = in_ch if wm == "all" else 2       # "none" also passes a dummy 2ch (ignored by the model; same as eval_echodiffusion)
        net = EchoDiffusionDepth(in_ch=in_ch, wave_mode="none" if wm == "none" else "cide",
                                 wave_ch=wave_ch, faithful=(args.get("port", "enhanced") == "faithful"))
        net.load_state_dict(ck["state_dict"]); net.to(device).eval()

        def fwd(b):
            return net(b["spec"].to(device), b["wave"].to(device)).float()

    print(f"[cfg] model={a.model} mode={mode} ckpt={a.ckpt} wave_mode={args.get('wave_mode')} md={md}", flush=True)
    out = {}
    for fam, yaws in FAMILIES.items():
        res = run_family(fwd, list(yaws), mode, wave_ch, device, md)
        out[fam] = res
        cols = KEYS + [b[0] for b in BANDS]
        print(f"[{fam}] n={res['n']} " + " ".join(f"{c}={res[c]:.4f}" for c in cols), flush=True)
    jp = os.path.join("comparison", f"rot30_eval_{a.model}_{mode}{a.tag}.json")
    json.dump(out, open(jp, "w"), indent=2)
    print(f"[done] -> {jp}", flush=True)


if __name__ == "__main__":
    main()
