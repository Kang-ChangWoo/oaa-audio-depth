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
    elif ABL["mode"] == "notch":
        w = w.clone(); w[:, int(ABL["m"] * SPM):int(ABL["m2"] * SPM)] = 0
    elif ABL["mode"] == "ampdirect":         # direct impulse amplitude x f, position preserved
        w = w.clone(); w[:, :200] = w[:, :200] * ABL["m"]
    elif ABL["mode"] == "shiftdirect":       # direct impulse shifted +k samples, amplitude preserved
        k = int(ABL["m"]); w2 = w.clone(); seg = w[:, :200].clone()
        w2[:, :200] = 0; w2[:, k:k+200] = w2[:, k:k+200] + seg; w = w2
    elif ABL["mode"] == "notch2":            # two notches: [m,m2] and [m3,m4] (metres)
        w = w.clone()
        w[:, int(ABL["m"]*SPM):int(ABL["m2"]*SPM)] = 0
        w[:, int(ABL["m3"]*SPM):int(ABL["m4"]*SPM)] = 0
    elif ABL["mode"] == "shuffle":
        i0 = int(ABL["m"] * SPM); blk = int(ABL.get("blk", 56))                 # 0.2 m blocks
        w = w.clone(); seg = w[:, i0:]
        n = seg.shape[1] // blk
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(n, generator=g)
        w[:, i0:i0 + n * blk] = seg[:, :n * blk].reshape(w.shape[0], n, blk)[:, perm].reshape(w.shape[0], -1)
    elif ABL["mode"] == "earlyzero_renorm":
        s0 = w.std()
        w = w.clone(); w[:, :int(ABL["m"] * SPM)] = 0
        w = w * (s0 / w.std().clamp(min=1e-8))            # restore overall energy statistics
    return _orig(w)

D._stft_mag = _masked

MODELS = {"cnn": "oaa_fb_fin", "eat_llrd": "0820_eatllrd_fb_rep", "sslam": "0820_sslam_fb_rep"}
import sys
SET = sys.argv[1] if len(sys.argv) > 1 else "v1"
if SET == "v1":
    CONDS = [("clean", 0.0), ("latecut", 6.0), ("latecut", 4.0), ("earlyzero", 1.0)]
    OUT = "comparison_0820/reverb_ablation.json"
elif SET == "sweep":
    CONDS = [("clean", 0.0)] + [("latecut", m) for m in (3.0, 5.0, 7.0, 8.0, 9.0)]
    OUT = "comparison_0820/reverb_ablation_sweep.json"
elif SET == "renorm":
    CONDS = [("clean", 0.0), ("earlyzero", 1.0), ("earlyzero_renorm", 1.0)]
    OUT = "comparison_0820/reverb_ablation_renorm.json"
elif SET == "shuffle":
    CONDS = [("clean", 0.0), ("shuffle", 6.0), ("latecut", 6.0)]
    OUT = "comparison_0820/reverb_ablation_shuffle.json"
elif SET == "seeds":
    MODELS = {"cnn_vw": "oaa_fb_vw", "sslam_s1": "0820_sslam_s1_fb_rep", "sslam_llrd": "0820_sslam_llrd_fb_rep"}
    CONDS = [("clean", 0.0), ("earlyzero", 1.0), ("latecut", 8.0)]
    OUT = "comparison_0820/reverb_ablation_seeds.json"
elif SET == "timing_amp":
    CONDS = [("clean", 0.0), ("ampdirect", 0.25), ("ampdirect", 4.0),
             ("shiftdirect", 58), ("shiftdirect", 116)]
    OUT = "comparison_0820/reverb_ablation_timing_amp.json"
elif SET == "blocksweep":
    CONDS = [("clean", 0.0)] + [("shuffleblk", b) for b in (6, 28, 140, 280)]
    OUT = "comparison_0820/reverb_ablation_blocksweep.json"
elif SET == "cross2x2":
    CONDS = [("notch", (3.0, 6.0)), ("notch2", (3.0, 6.0, 8.0, 10.0))]
    OUT = "comparison_0820/reverb_ablation_cross2x2.json"
elif SET == "seedmech":
    MODELS = {"cnn_vw": "oaa_fb_vw", "sslam_s1": "0820_sslam_s1_fb_rep", "sslam_llrd": "0820_sslam_llrd_fb_rep"}
    CONDS = [("clean", 0.0), ("latecut", 6.0), ("shuffle", 6.0)]
    OUT = "comparison_0820/reverb_ablation_seedmech.json"
elif SET == "notch":
    CONDS = [("clean", 0.0), ("notch", (2.0, 4.0)), ("notch", (4.0, 6.0)),
             ("notch", (6.0, 8.0)), ("notch", (8.0, 10.0))]
    OUT = "comparison_0820/reverb_ablation_notch.json"

def main():
    DM = get_data_module()
    dev = torch.device("cuda")
    out = {}
    for mode, m in CONDS:
        if mode == "shuffleblk":
            ABL["mode"], ABL["m"], ABL["blk"] = "shuffle", 6.0, m
            cname = f"shuffle6m_blk{m}smp"
        elif mode == "notch2":
            ABL["mode"] = mode; ABL["m"], ABL["m2"], ABL["m3"], ABL["m4"] = m
            cname = f"notch{m[0]:g}-{m[1]:g}+{m[2]:g}-{m[3]:g}m"
        elif mode == "notch":
            ABL["mode"], ABL["m"], ABL["m2"] = mode, m[0], m[1]
            cname = f"notch{m[0]:g}-{m[1]:g}m"
        else:
            ABL["mode"], ABL["m"] = mode, m
            ABL["blk"] = 56
            cname = mode if mode == "clean" else f"{mode}{m:g}" + ("smp" if mode == "shiftdirect" else "x" if mode == "ampdirect" else "m")
        out[cname] = {}
        for tag, run in MODELS.items():
            rd = resolve_run(run, ["comparison", "comparison_0820"])
            r = evaluate(rd, DM, "best", dev)
            out[cname][tag] = {k: round(float(v), 4) for k, v in r.items()}
            print(f"[{cname:12}] {tag:9} MAE={r['MAE']:.4f} near={r['near<3']:.4f} "
                  f"mid={r['mid3-6']:.4f} far={r['far>6']:.4f}", flush=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n== deltas vs clean (MAE / near / mid / far) ==")
    for cname in out:
        if cname == "clean": continue
        for tag in MODELS:
            c, a = out["clean"][tag], out[cname][tag]
            print(f"{cname:12} {tag:9} d=" + " / ".join(
                f"{a[k]-c[k]:+.4f}" for k in ("MAE", "near<3", "mid3-6", "far>6")))
    print(f"[saved] {OUT}", flush=True)

if __name__ == "__main__":
    main()
