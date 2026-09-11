# Mic-drop degradation: 8 mics at train time, 1-8 at test time

Replica test (off3), MAE in metres. Channels are zeroed and the pose conditioning stays
truthful, so this is the *mic failure* curve, not a smaller-rig retrain. k live mics is the
mean over 3 fixed-seed random subsets. `ret@1` = MAE(1 mic) / MAE(8 mic): how much of the
8-mic error the model keeps when only one mic survives (lower is a flatter, more graceful
curve).

| model | 8mic | 7mic | 6mic | 5mic | 4mic | 3mic | 2mic | 1mic | ret@1 |
|---|---|---|---|---|---|---|---|---|---|
| **ours, no mic-drop training** | | | | | | | | | |
| OAA-CNN (ours) | 0.2668 | 0.3386 | 0.4854 | 0.5978 | 0.7135 | 0.8521 | 0.9629 | 1.0904 | 4.09x |
| eat+LLRD (ours) | 0.2535 | 0.4131 | 0.5811 | 0.6547 | 0.6832 | 0.7118 | 0.7530 | 0.7937 | 3.13x |
| **prior work (no mic-drop by construction)** | | | | | | | | | |
| BatVision | 0.2732 | 0.5709 | 0.5319 | 0.7604 | 0.8195 | 0.9185 | 0.9420 | 1.1059 | 4.05x |

_pending: 0820_sslam_r8_rep, es_r8_fin, 0820_beyond_r8_rep, rn_r8_fin, vit_r8_fin, oaa_r8_fin, oaa_r8_kany_
