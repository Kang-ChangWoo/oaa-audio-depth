"""Roster for the 8->1 mic degradation study (Replica, off3 test split).

Every entry is an 8-channel (r8) model. The study question is what happens when mics die at
inference time on a rig that was trained with all eight: we zero k channels (k=0..7), keep the
pose conditioning truthful, and read the curve from 8 live mics down to 1.

GROUPS
  novd   trained WITHOUT mic drop -- the honest "trained at 8, deployed degraded" case, and the
         only setting in which our encoders and the prior-work baselines are directly comparable
         (no prior-work model uses mic-drop augmentation).
  prior  published baselines, all no-mic-drop by construction.
  vdrop  the same architectures trained WITH mic drop (CNN k<=4 released and k<=6 "kany"; the
         0820 AFM variants at k<=4). Not part of the like-for-like comparison against prior work,
         but paired with novd it prices what the augmentation buys, per encoder family.
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
    # vdrop group: the SAME architectures re-trained with mic-drop augmentation. Paired with the
    # novd entries above this makes a {CNN, AFM} x {no mic-drop, mic-drop} 2x2 — the only way to
    # separate "which encoder degrades gracefully" from "which encoder was taught to".
    ("OAA-CNN +vd(k<=4)",   "vdrop", "oaa_r8_fin"),
    ("OAA-CNN +vd(k<=6)",   "vdrop", "oaa_r8_kany"),
    ("eat+LLRD +vd",        "vdrop", "0820_eatllrd_r8_rep"),
    ("eat+LLRD+cs +vd",     "vdrop", "0820_eatllrd_cs_r8_rep"),
    ("sslam +vd",           "vdrop", "0820_sslam_r8vd_rep"),
    ("sslam+LLRD +vd",      "vdrop", "0820_sslam_llrd_r8vd_rep"),
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
