"""Roster for the 8->1 mic degradation study (Replica, off3 test split).

Every entry is an 8-channel (r8) model. The study question is what happens when mics die at
inference time on a rig that was trained with all eight: we zero k channels (k=0..7), keep the
pose conditioning truthful, and read the curve from 8 live mics down to 1.

GROUPS
  novd   trained WITHOUT mic drop -- the honest "trained at 8, deployed degraded" case, and the
         only setting in which our encoders and the prior-work baselines are directly comparable
         (no prior-work model uses mic-drop augmentation).
  prior  published baselines, all no-mic-drop by construction.
  vdrop  reference only: OAA-CNN trained WITH mic drop (k<=4 released, k<=6 "kany"), included to
         price what the augmentation buys. NOT part of the like-for-like comparison.
"""

ROSTER = [
    # label,                group,   run-name
    ("OAA-CNN (ours)",      "novd",  "oaa_r8_novdrop"),
    ("eat+LLRD (ours)",     "novd",  "0820_eatllrd_r8novd_rep"),
    ("sslam (ours)",        "novd",  "0820_sslam_r8_rep"),
    ("BatVision",           "prior", "bat_r8_fin"),
    ("EchoScan",            "prior", "es_r8_fin"),
    ("Beyond-I2D",          "prior", "0820_beyond_r8_rep"),
    ("ResNet-18",           "prior", "rn_r8_fin"),
    ("ViT-B",               "prior", "vit_r8_fin"),
    ("OAA-CNN +vdrop(k<=4)", "vdrop", "oaa_r8_fin"),
    ("OAA-CNN +vdrop(k<=6)", "vdrop", "oaa_r8_kany"),
]

# EchoDiffusion needs its isolated env, so it runs through analysis/micdrop_eco.py
# ($ECHODIFF_PY) and is merged into the same JSON by key.
ECO = ("EchoDiffusion", "prior", "eco_r8_fin")

GROUP_ORDER = ["novd", "prior", "vdrop"]
GROUP_LABEL = {
    "novd":  "ours, no mic-drop training",
    "prior": "prior work (no mic-drop by construction)",
    "vdrop": "reference: mic-drop trained",
}
