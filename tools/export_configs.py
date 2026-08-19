"""Export the exact training arguments of the final runs to configs/<dataset>/<run>.json.

Reads <run>/train_done.json["args"] for every run listed (default: all *_fin runs in comparison/
(Replica) and the FIN_SELECTION.md rows in comparison_mp3d/ (Matterport3D)), so the per-run recipe
behind every reported number is recorded in a small, versioned file.
Run:  python tools/export_configs.py
"""
import os, sys, json, glob, re
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETS = {"replica": ("comparison", "data_0422"), "mp3d": ("comparison_mp3d", "data_mp3d")}
# Matterport3D final EchoDiffusion dirs are not suffixed _fin (see comparison_mp3d/FIN_SELECTION.md)
MP3D_EXTRA = ["eco_r2_wstd", "eco_fb_wstd", "eco_r6", "eco_r8"]
DROP = {"run_name", "out_dir", "num_workers", "resume", "cide_cache"}   # machine/bookkeeping only


def export(ds, cdir, dm):
    runs = sorted(os.path.basename(d) for d in glob.glob(f"{HERE}/{cdir}/*_fin"))
    if ds == "mp3d":
        runs += [r for r in MP3D_EXTRA if os.path.isdir(f"{HERE}/{cdir}/{r}")]
    out = f"{HERE}/configs/{ds}"; os.makedirs(out, exist_ok=True); n = 0
    for r in runs:
        td = f"{HERE}/{cdir}/{r}/train_done.json"
        if not os.path.isfile(td):
            print(f"[skip] {cdir}/{r}: no train_done.json"); continue
        j = json.load(open(td)); a = {k: v for k, v in j.get("args", {}).items() if k not in DROP}
        rec = {"run": r, "dataset": ds, "DATA_MODULE": dm, "best_val_mae_m": j.get("best_val_mae_m"),
               "epochs_trained": len(j.get("hist", [])) or None, "args": a}
        json.dump(rec, open(f"{out}/{r}.json", "w"), indent=1); n += 1
    print(f"[{ds}] {n} configs -> configs/{ds}/")


if __name__ == "__main__":
    for ds, (cdir, dm) in SETS.items():
        if os.path.isdir(f"{HERE}/{cdir}"):
            export(ds, cdir, dm)
