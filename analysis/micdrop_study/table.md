# Mic-drop degradation: every model trained on 8 mics, tested on 8-1

Replica test (off3), MAE in metres. At inference k of the 8 channels are zeroed and the
pose conditioning stays truthful -- the *mic failure* curve, not a smaller-rig retrain.
Each k is the mean over 3 fixed-seed random subsets. `mic-drop` = did the model see
dropped mics during TRAINING (among prior work only EchoDiffusion has such a variant).
`ret@1` = MAE(1 mic)/MAE(8 mic): how much of the 8-mic error survives total rig loss --
lower is a flatter, more graceful curve. Sorted by 1-mic MAE.

| model | origin | mic-drop | 8mic | 7mic | 6mic | 5mic | 4mic | 3mic | 2mic | 1mic | ret@1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OAA-CNN +vd(k<=6) | ours | yes | **0.2336** | **0.2358** | **0.2379** | **0.2593** | **0.2733** | **0.2856** | **0.3290** | **0.4509** | 1.93x |
| OAA-CNN +vd(k<=4) | ours | yes | 0.2368 | 0.2394 | 0.2413 | 0.2594 | 0.2758 | 0.2931 | 0.3608 | 0.5272 | 2.23x |
| eat+LLRD | ours | no | 0.2535 | 0.4131 | 0.5811 | 0.6547 | 0.6832 | 0.7118 | 0.7530 | 0.7937 | 3.13x |
| sslam | ours | no | 0.2399 | 0.3675 | 0.4482 | 0.5156 | 0.5827 | 0.6279 | 0.7249 | 0.7995 | 3.33x |
| EchoDiffusion +chdrop | prior work | yes | 0.2722 | 0.2798 | 0.2915 | 0.3760 | 0.4069 | 0.4589 | 0.6488 | 0.8826 | 3.24x |
| EchoDiffusion | prior work | no | 0.2600 | 0.3859 | 0.4696 | 0.6646 | 0.7053 | 0.7827 | 0.7795 | 0.9221 | 3.55x |
| OAA-CNN | ours | no | 0.2668 | 0.3386 | 0.4854 | 0.5978 | 0.7135 | 0.8521 | 0.9629 | 1.0904 | 4.09x |
| BatVision | prior work | no | 0.2732 | 0.5709 | 0.5319 | 0.7604 | 0.8195 | 0.9185 | 0.9420 | 1.1059 | 4.05x |
| EchoScan | prior work | no | 0.3400 | 0.5269 | 0.6678 | 0.8649 | 1.0267 | 1.2946 | 1.5283 | 1.4131 | 4.16x |

_pending: 0820_eatllrd_r8_rep, 0820_eatllrd_cs_r8_rep, 0820_sslam_r8vd_rep, 0820_sslam_llrd_r8vd_rep, 0820_beyond_r8_rep, rn_r8_fin, vit_r8_fin_
