import json, sys, os

KEYS = ("cue", "loss_weight", "epochs", "nviews", "data_mode", "lr", "warmup_ep",
        "batch_size", "accum", "seed", "subset_aug", "vdrop_kmax", "vdrop_p",
        "vdrop_ramp", "wd", "max_depth", "in_ch", "run_name")

for d in sys.argv[1:]:
    p = os.path.join(d, "train_done.json")
    if not os.path.exists(p):
        print(d, "MISSING")
        continue
    j = json.load(open(p))
    a = j["args"]
    hist = j["hist"]
    best = j["best_val_mae_m"]
    bep = min(hist, key=lambda h: h["val_mae_m"])["epoch"]
    last5 = any(h["val_mae_m"] <= best + 1e-12 for h in hist[-5:])
    print("---", d)
    print("  best=%.4f best_ep=%d n_ep=%d last5_improved=%s" % (best, bep, len(hist), last5))
    print("  " + json.dumps({k: a[k] for k in KEYS if k in a}, sort_keys=True))
