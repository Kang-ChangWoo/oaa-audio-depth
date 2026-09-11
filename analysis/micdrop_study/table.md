# Mic-drop degradation: every model trained on 8 mics, tested on 8-1

Replica test (off3), MAE in metres. At inference k of the 8 channels are zeroed and the
pose conditioning stays truthful — the *mic failure* curve, not a smaller-rig retrain.
Each k is the mean over 3 fixed-seed random subsets. `mic-drop` says whether the model saw
dropped mics during TRAINING (no prior-work model does). `ret@1` = MAE(1 mic)/MAE(8 mic):
how much of the 8-mic error survives total rig loss — lower is a flatter, more graceful
curve. Sorted by 1-mic MAE (worst case first-class, since that is what the study is about).

| model | origin | mic-drop | 8mic | 7mic | 6mic | 5mic | 4mic | 3mic | 2mic | 1mic | ret@1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eat+LLRD (ours) | ours | no | **0.2535** | 0.4131 | 0.5811 | 0.6547 | **0.6832** | **0.7118** | **0.7530** | **0.7937** | 3.13x |
| OAA-CNN (ours) | ours | no | 0.2668 | **0.3386** | **0.4854** | **0.5978** | 0.7135 | 0.8521 | 0.9629 | 1.0904 | 4.09x |
| BatVision | prior work | no | 0.2732 | 0.5709 | 0.5319 | 0.7604 | 0.8195 | 0.9185 | 0.9420 | 1.1059 | 4.05x |
| EchoScan | prior work | no | 0.3400 | 0.5269 | 0.6678 | 0.8649 | 1.0267 | 1.2946 | 1.5283 | 1.4131 | 4.16x |

_pending: 0820_sslam_r8_rep, 0820_beyond_r8_rep, rn_r8_fin, vit_r8_fin, oaa_r8_fin, oaa_r8_kany, 0820_eatllrd_r8_rep, 0820_eatllrd_cs_r8_rep, 0820_sslam_r8vd_rep, 0820_sslam_llrd_r8vd_rep_
