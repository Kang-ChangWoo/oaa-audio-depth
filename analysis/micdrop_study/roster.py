"""Roster for the 8->1 mic degradation study (Replica, off3 test split).

Every entry is an 8-channel (r8) model. The question is what happens when mics die at inference
on a rig trained with all eight: we zero k channels (k=0..7), keep the pose conditioning
truthful, and read the curve from 8 live mics down to 1.

Each entry is (label, origin, trained_with_mic_drop, run_name).

  origin                 "ours" (this line of work) or "prior" (published baseline)
  trained_with_mic_drop  whether the model saw dropped mics during TRAINING. This is the axis
                         that matters: no published baseline ships such an augmentation except
                         EchoDiffusion's own channel dropout, so the flag separates "degrades
                         gracefully by construction" from "was taught to".

Curves already computed for the release live in comparison/eval_micdrop.json (RELEASED);
report.py folds them in so the study covers every method we have a curve for. A shard re-run
here takes precedence over the released copy of the same run.
"""

# label,                    origin,  mic-drop,  run-name
ROSTER = [
    # -- ours, no mic-drop augmentation: the like-for-like setting against prior work
    ("OAA-CNN",              "ours",  False, "oaa_r8_novdrop"),
    ("eat+LLRD",             "ours",  False, "0820_eatllrd_r8novd_rep"),
    ("sslam",                "ours",  False, "0820_sslam_r8_rep"),
    # -- ours, trained WITH mic drop
    ("OAA-CNN +vd(k<=4)",    "ours",  True,  "oaa_r8_fin"),
    ("OAA-CNN +vd(k<=6)",    "ours",  True,  "oaa_r8_kany"),
    ("eat+LLRD +vd",         "ours",  True,  "0820_eatllrd_r8_rep"),
    ("eat+LLRD+cs +vd",      "ours",  True,  "0820_eatllrd_cs_r8_rep"),
    ("sslam +vd",            "ours",  True,  "0820_sslam_r8vd_rep"),
    ("sslam+LLRD +vd",       "ours",  True,  "0820_sslam_llrd_r8vd_rep"),
    # -- published baselines
    ("BatVision",            "prior", False, "bat_r8_fin"),
    ("EchoScan",             "prior", False, "es_r8_fin"),
    ("Beyond-I2D",           "prior", False, "0820_beyond_r8_rep"),
    ("ResNet-18",            "prior", False, "rn_r8_fin"),
    ("ViT-B",                "prior", False, "vit_r8_fin"),
]

# EchoDiffusion runs in its isolated env via analysis/micdrop_eco.py ($ECHODIFF_PY).
ECO = ("EchoDiffusion", "prior", False, "eco_r8_fin")

# Released curves (identical protocol). eco_r8_chdrop is EchoDiffusion retrained with its own
# channel dropout -- the only prior-work model that has a mic-drop-trained variant at all.
RELEASED = "comparison/eval_micdrop.json"
RELEASED_ENTRIES = [
    ("EchoDiffusion",         "prior", False, "eco_r8_fin"),
    ("EchoDiffusion +chdrop", "prior", True,  "eco_r8_chdrop"),
]
