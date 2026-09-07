"""0820 mechanism experiment: late-reverb / early-arrival ablation at TEST TIME (no training).

Hypothesis (H): SSLAM's advantage comes from *distributed late-reverberant structure*
(mixture-SSL preserves it), while the task CNN relies on *local early arrival-time* cues.
Predictions:  latecut  -> dMAE(sslam) >> dMAE(cnn), concentrated in mid/far;
              earlyzero-> dMAE(cnn)   >  dMAE(sslam), concentrated in near/mid.

Conditions (waveform time masks, applied inside data_0422._stft_mag; spec cache disabled):
  clean          - sanity (must reproduce compare.json rows)
  latecut6m      - zero t > 6 m round-trip (1679 smp): keeps primary echoes of ALL <6 m surfaces,
                   removes far-surface primaries AND every multi-bounce late tail
  latecut4m      - harsher (1120 smp)
  earlyzero1m    - zero t < 1 m round-trip (280 smp): removes direct sound + rig-proximal onsets,
                   keeps the full reverberant tail

Run: REPLICA_ROOT=... R0422_SPLIT=off3 DATA_MODULE=data_0422 EVAL_BS=8 \
     python 0820_reverb_ablation.py   -> comparison_0820/reverb_ablation.json
"""
import os, json
os.environ.pop("REPLICA_SPEC_CACHE", None)          # force on-the-fly STFT so the mask applies
os.environ.setdefault("DATA_MODULE", "data_0422")
import torch
import data_0422 as D
from core.data import get_data_module
from core.ckpt import resolve_run
from core.evaluate import evaluate

SPM = 279.9                                          # samples per metre (round-trip, 48 kHz)
ABL = {"mode": "clean", "m": 0.0}
_orig = D._stft_mag

def _masked(wav2):
    w = wav2
    if ABL["mode"] == "latecut":
        w = w.clone(); w[:, int(ABL["m"] * SPM):] = 0
    elif ABL["mode"] == "earlyzero":
        w = w.clone(); w[:, :int(ABL["m"] * SPM)] = 0
    return _orig(w)

D._stft_mag = _masked

MODELS = {"cnn": "oaa_fb_fin", "eat_llrd": "0820_eatllrd_fb_rep", "sslam": "0820_sslam_fb_rep"}
CONDS = [("clean", 0.0), ("latecut", 6.0), ("latecut", 4.0), ("earlyzero", 1.0)]

def main():
    DM = get_data_module()
    dev = torch.device("cuda")
    out = {}
    for mode, m in CONDS:
        ABL["mode"], ABL["m"] = mode, m
        cname = mode if mode == "clean" else f"{mode}{m:g}m"
        out[cname] = {}
        for tag, run in MODELS.items():
            rd = resolve_run(run, ["comparison", "comparison_0820"])
            r = evaluate(rd, DM, "best", dev)
            out[cname][tag] = {k: round(float(v), 4) for k, v in r.items()}
            print(f"[{cname:12}] {tag:9} MAE={r['MAE']:.4f} near={r['near<3']:.4f} "
                  f"mid={r['mid3-6']:.4f} far={r['far>6']:.4f}", flush=True)
    json.dump(out, open("comparison_0820/reverb_ablation.json", "w"), indent=2)
    print("\n== deltas vs clean (MAE / near / mid / far) ==")
    for cname in out:
        if cname == "clean": continue
        for tag in MODELS:
            c, a = out["clean"][tag], out[cname][tag]
            print(f"{cname:12} {tag:9} d=" + " / ".join(
                f"{a[k]-c[k]:+.4f}" for k in ("MAE", "near<3", "mid3-6", "far>6")))
    print("[saved] comparison_0820/reverb_ablation.json", flush=True)

if __name__ == "__main__":
    main()
