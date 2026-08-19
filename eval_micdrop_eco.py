"""Progressive mic-drop curve for EchoDiffusion (isolated env, fp32).

Same protocol as eval_micdrop.py (k of 8 spec channels zeroed, 3 fixed-seed draws per k).
The CIDE waveform pair corresponds to spec channels 0/1 (front L/R): a dropped front mic
also zeroes its waveform channel — a dead mic yields neither spec nor wave.

  CUDA_VISIBLE_DEVICES=? DATA_MODULE=data_0422 R0422_SPLIT=off3 HF_HOME=... \
    <echodiff_env>/bin/python eval_micdrop_eco.py --run-name eco_r8_fin
"""
import os, json, math, argparse, random, importlib
import torch
from model.echodiffusion import EchoDiffusionDepth

_DM = importlib.import_module(os.environ.get("DATA_MODULE", "data_0422"))
DRAWS = 3


def cos_lat(h, device):
    v = torch.arange(h, device=device, dtype=torch.float32)
    return torch.cos((math.pi / 2) - (v + 0.5) / h * math.pi).clamp(min=1e-3)


def variants_for(nch):
    out = [("k0", None)]
    rng = random.Random(0)                      # same seed as eval_micdrop.py -> same subsets
    for k in range(1, nch):
        for d in range(DRAWS):
            out.append((f"k{k}_d{d}", tuple(sorted(rng.sample(range(nch), k)))))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", nargs="+", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--out", default="comparison/eval_micdrop.json")
    a = ap.parse_args()
    device = torch.device("cuda")
    saved = {}
    if os.path.exists(a.out):
        try: saved = json.load(open(a.out))
        except Exception: saved = {}
    for run in a.run_name:
        rd = os.path.join("comparison", run)
        ck = torch.load(os.path.join(rd, f"{a.ckpt}.pth"), map_location="cpu", weights_only=False)
        args = ck["args"]; mode = args.get("mode", "r8"); md = args.get("max_depth", 10.0)
        in_ch = _DM.IN_CH[mode]
        wm = args.get("wave_mode", "std")
        wch = in_ch if wm == "all" else 2
        model = EchoDiffusionDepth(in_ch=in_ch, wave_mode="none" if wm == "none" else "cide",
                                   wave_ch=wch, faithful=(args.get("port", "enhanced") == "faithful")).to(device)
        model.load_state_dict(ck["state_dict"]); model.eval()
        vs = variants_for(in_ch)
        ld = _DM.spec_wave_loader("test", 8, False, 5, mode)
        wlat = cos_lat(256, device).view(1, 1, 256, 1)
        MK = ["MAE", "RMSE", "AbsRel", "log10", "delta1", "delta2", "delta3"]
        acc = {t: dict({k: 0.0 for k in MK}, n=0) for t, _ in vs}
        for b in ld:
            x0 = b["spec"].to(device); w0 = b["wave"][:, :wch].to(device)
            gt = b["depth"].to(device) * md; mask = b["mask"].to(device)
            w = wlat * mask; B = x0.shape[0]
            pi = lambda num, den: (num.flatten(1).sum(1) / den.flatten(1).sum(1).clamp(min=1e-6))
            for tag, idx in vs:
                x, wv = x0, w0
                if idx:
                    x = x0.clone(); x[:, list(idx)] = 0
                    dead = [j for j in idx if j < wch]
                    if dead:
                        wv = w0.clone(); wv[:, dead] = 0
                D = model(x, wv).float() * md
                a2 = acc[tag]
                a2["MAE"] += float(pi((D - gt).abs() * w, w).mean()) * B
                a2["RMSE"] += float(pi(((D - gt) ** 2) * w, w).clamp(min=0).sqrt().mean()) * B
                a2["AbsRel"] += float(pi((D - gt).abs() / gt.clamp(min=0.1) * w, w).mean()) * B
                a2["log10"] += float(pi((torch.log10(D.clamp(min=0.1)) - torch.log10(gt.clamp(min=0.1))).abs() * w, w).mean()) * B
                rt = torch.maximum(D.clamp(min=0.1) / gt.clamp(min=0.1), gt.clamp(min=0.1) / D.clamp(min=0.1))
                for i, dk in enumerate(["delta1", "delta2", "delta3"], 1):
                    a2[dk] += float(pi((rt < 1.25 ** i).float() * w, w).mean()) * B
                a2["n"] += B
        res = {t: acc[t]["MAE"] / acc[t]["n"] for t, _ in vs}
        res_full = {t: {k: acc[t][k] / acc[t]["n"] for k in MK} for t, _ in vs}
        curve = {"k0": res["k0"]}
        for k in range(1, in_ch):
            ds = [res[f"k{k}_d{d}"] for d in range(DRAWS)]
            curve[f"k{k}"] = {"mean": sum(ds) / len(ds), "min": min(ds), "max": max(ds),
                              "draws": {f"d{d}": ds[d] for d in range(DRAWS)}}
        curve_full = {"k0": res_full["k0"]}
        for k in range(1, in_ch):
            curve_full[f"k{k}"] = {m: sum(res_full[f"k{k}_d{d}"][m] for d in range(DRAWS)) / DRAWS for m in MK}
        saved[run] = {"curve": curve, "curve_full": curve_full, "subsets": {t: list(i) for t, i in vs if i}}
        json.dump(saved, open(a.out, "w"), indent=2)
        print(f"== {run} (MAE, remaining mics = {in_ch}-k)")
        print(f"  k0({in_ch}mic): {res['k0']:.4f}")
        for k in range(1, in_ch):
            c = curve[f"k{k}"]
            print(f"  k{k}({in_ch-k}mic): mean {c['mean']:.4f}  [{c['min']:.4f}~{c['max']:.4f}]", flush=True)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
