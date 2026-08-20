"""0820 screening summary: comparison_0820/*/train_done.json (+ optional compare.json) -> one table.

  python 0820_collect.py             # val-MAE progress table of every 0820_ run
  python 0820_collect.py --test      # also merge test metrics from comparison_0820/compare.json
Reference rows: the CNN finals (comparison/oaa_fb_fin, comparison_mp3d/oaa_fb_fin).
"""
import argparse, glob, json, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="comparison_0820")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    rows = []
    for td in sorted(glob.glob(f"{a.dir}/*/train_done.json")):
        j = json.load(open(td)); ar = j["args"]
        rows.append((os.path.basename(os.path.dirname(td)), ar.get("audio_backbone", "?"),
                     ar.get("data_module", "?"), len(j.get("hist", [])), ar.get("epochs"),
                     j.get("best_val_mae_m")))
    for run in sorted(glob.glob(f"{a.dir}/0820_*/")):                 # in-progress runs (last.pth only)
        if not os.path.exists(os.path.join(run, "train_done.json")) and os.path.exists(os.path.join(run, "last.pth")):
            import torch
            ck = torch.load(os.path.join(run, "last.pth"), map_location="cpu", weights_only=False)
            ar = ck["args"]
            rows.append((os.path.basename(run.rstrip("/")) + " (running)", ar.get("audio_backbone", "?"),
                         ar.get("data_module", "?"), ck.get("next_epoch"), ar.get("epochs"), None))
    for ref, ds in (("comparison/oaa_fb_fin", "data_0422"), ("comparison_mp3d/oaa_fb_fin", "data_mp3d")):
        td = os.path.join(ref, "train_done.json")
        if os.path.exists(td):
            j = json.load(open(td))
            rows.append((ref + " [CNN ref]", "cnn", ds, len(j.get("hist", [])), j["args"].get("epochs"),
                         j.get("best_val_mae_m")))
    print(f"{'run':42} {'backbone':12} {'dataset':10} {'ep':>7} {'best_val_MAE_m':>15}")
    for r in rows:
        print(f"{r[0]:42} {r[1]:12} {r[2]:10} {str(r[3])+'/'+str(r[4]):>7} "
              f"{('%.4f' % r[5]) if r[5] else '-':>15}")
    if a.test:
        for cj in (f"{a.dir}/compare.json",):
            if os.path.exists(cj):
                d = json.load(open(cj))
                print(f"\n== test ({cj}) ==")
                ks = ["MAE", "RMSE", "AbsRel", "delta1", "Params(M)"]
                print(f"{'run':42}" + "".join(f"{k:>10}" for k in ks))
                for r, v in sorted(d.items()):
                    print(f"{r:42}" + "".join(f"{v[k]:10.4f}" for k in ks))


if __name__ == "__main__":
    main()
