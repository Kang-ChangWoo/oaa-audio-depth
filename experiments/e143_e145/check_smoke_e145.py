#!/usr/bin/env python3
"""E145 smoke check: verify the two-checkpoint mechanism before the nine 80-epoch runs.

Exits non-zero on any failure -- launch_e145.sh must not be run until this passes.
usage: check_smoke_e145.py <ctrl_run_dir> <invfreq_run_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ckpt_content_hash import content_hash


def sha(p):
    """CONTENT hash, not file hash. torch.save names the zip container after the output filename
    (best.pth/data.pkl vs best_own.pth/data.pkl), so two checkpoints holding identical weights can
    never be byte-identical. See ckpt_content_hash.py."""
    return content_hash(p)[0]


def load(rd):
    p = os.path.join(rd, "train_done.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def main():
    ctrl_d, inv_d = sys.argv[1], sys.argv[2]
    fails, notes = [], []

    for rd in (ctrl_d, inv_d):
        for f in ("best.pth", "best_own.pth", "train_done.json"):
            if not os.path.exists(os.path.join(rd, f)):
                fails.append(f"MISSING {rd}/{f}")

    c, i = load(ctrl_d), load(inv_d)
    if not c or not i:
        print("\n".join(fails) or "train_done.json missing")
        sys.exit(1)

    # ---- CHECK 1: ctrl, loss-weight none -> the two val metrics must be bit-identical ----
    bad = [h["epoch"] for h in c["hist"] if h["val_obj_m"] != h["val_mae_m"]]
    if bad:
        fails.append(f"CHECK1 FAIL ctrl: val_obj_m != val_mae_m at epochs {bad} "
                     "(weight_fn('none') must return exactly 1.0)")
    else:
        notes.append(f"CHECK1 OK  ctrl: val_obj_m == val_mae_m bit-for-bit, all {len(c['hist'])} epochs")

    # ---- CHECK 2: ctrl -> best_own.pth must be BYTE-IDENTICAL to best.pth (PREREG s4) ----
    if os.path.exists(os.path.join(ctrl_d, "best.pth")) and os.path.exists(os.path.join(ctrl_d, "best_own.pth")):
        h1, h2 = sha(os.path.join(ctrl_d, "best.pth")), sha(os.path.join(ctrl_d, "best_own.pth"))
        if h1 != h2:
            fails.append(f"CHECK2 FAIL ctrl: content hash best.pth {h1[:16]} != best_own.pth {h2[:16]}")
        else:
            notes.append(f"CHECK2 OK  ctrl: best.pth and best_own.pth content-identical ({h1[:16]}...)")

    # ---- CHECK 3: invfreq -> the two val metrics must actually differ (mechanism is live) ----
    same = [h["epoch"] for h in i["hist"] if h["val_obj_m"] == h["val_mae_m"]]
    if same:
        fails.append(f"CHECK3 FAIL invfreq: val_obj_m == val_mae_m at epochs {same} "
                     "(the arm weight is not reaching the val metric)")
    else:
        notes.append("CHECK3 OK  invfreq: val_obj_m differs from val_mae_m every epoch; "
                     + ", ".join(f"ep{h['epoch']} mae={h['val_mae_m']:.4f} obj={h['val_obj_m']:.4f}"
                                 for h in i["hist"]))

    # ---- CHECK 4: the two selection indices are tracked independently ----
    for tag, j in (("ctrl", c), ("invfreq", i)):
        hist = j["hist"]
        em = min(hist, key=lambda h: h["val_mae_m"])["epoch"]
        eo = min(hist, key=lambda h: h["val_obj_m"])["epoch"]
        if j.get("best_ep_mae") != em or j.get("best_ep_obj") != eo:
            fails.append(f"CHECK4 FAIL {tag}: train_done says mae@{j.get('best_ep_mae')} "
                         f"obj@{j.get('best_ep_obj')} but hist argmins are {em} / {eo}")
        else:
            notes.append(f"CHECK4 OK  {tag}: best_ep_mae={em} best_ep_obj={eo} (agree with hist)")

    # ---- CHECK 5: invfreq file divergence -- INFORMATIONAL, not a gate ----
    if os.path.exists(os.path.join(inv_d, "best_own.pth")):
        h1, h2 = sha(os.path.join(inv_d, "best.pth")), sha(os.path.join(inv_d, "best_own.pth"))
        em = min(i["hist"], key=lambda h: h["val_mae_m"])["epoch"]
        eo = min(i["hist"], key=lambda h: h["val_obj_m"])["epoch"]
        if em == eo:
            notes.append(f"CHECK5 INFO invfreq: both criteria pick ep{em} in this short smoke, so the "
                         f"files coincide ({h1[:16]}...). Expected at 4 epochs; file-level divergence "
                         "can only show once the two curves separate. Not a failure.")
            if h1 != h2:
                fails.append("CHECK5 FAIL invfreq: argmins agree but the files differ -- "
                             "the checkpoints are not the EMA snapshot they claim to be (content hash)")
        else:
            if h1 == h2:
                fails.append(f"CHECK5 FAIL invfreq: argmins differ (mae@{em} vs obj@{eo}) but the "
                             "files are identical -- best_own.pth is not being written")
            else:
                notes.append(f"CHECK5 OK  invfreq: argmins differ (mae@{em} vs obj@{eo}) and the "
                             "checkpoints differ -> full mechanism demonstrated end-to-end")

    # ---- CHECK 6: diagnostic third metric is present ----
    for tag, j in (("ctrl", c), ("invfreq", i)):
        if not all("val_trainform" in h for h in j["hist"]):
            fails.append(f"CHECK6 FAIL {tag}: val_trainform missing from hist")
    if not any(f.startswith("CHECK6") for f in fails):
        notes.append("CHECK6 OK  val_trainform logged for both runs (diagnostic, selects nothing)")

    print("\n".join(notes))
    if fails:
        print("\n".join("  !! " + f for f in fails))
        print("SMOKE RESULT: FAIL")
        sys.exit(1)
    print("SMOKE RESULT: PASS")


if __name__ == "__main__":
    main()
