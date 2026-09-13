#!/bin/bash
# 0820 Stage-4a: the "Differential SSLAM-OAA" programme, stage 1 (stem search) + leftovers.
#
# Baseline for everything here is sslam + LLRD 0.75 + convstem (A0), the strongest AFM
# configuration in the campaign. Two axes are opened, both zero-initialised so every variant
# STARTS bit-identical to A0 (verified: max|dy| = 0 at init):
#
#  A. patch-stem variants (model/audio_backbones_0820.py::_STEMS). A0 = "conv" (4x stride-2 3x3).
#     A1 conv_res   stride-2 3x5 entry + stride-1 residual 3x5 pair (time > freq receptive field)
#     A2 conv_ms    two branches after one stride-2: plain 3x3 + temporally dilated 3x3, concat+1x1
#     A3 conv_fact  factorised 3x1 frequency conv then 1x5 temporal conv
#     stem params +5.8% / +3.8% / +0.9% over A0 (<10% budget); whole-model FLOPs +4.1/+2.7/+0.7%.
#     Cells: the three hard ones from the plan -- Replica r8, MP3D r6, MP3D r8 (the collapse cell).
#     Settings are copied verbatim from the matching A0 run so the stem is the ONLY difference.
#
#  B. early mic-differential probe (2 runs, A0 stem). D_i = S_i - mean_j S_j injected as a
#     channel-wise gated residual; gate bias -2 (sigmoid~0.12) and P zero-init. Run on the two MP3D
#     cells now rather than after stage 1, because the stem axis and the fusion axis are
#     independent and MP3D 6/8ch is where the decision is. Explicitly an early probe, not the
#     stage-2 grid -- that runs on the winning stem.
#
#  E. the two stage-3r jobs the previous dispatcher never reached (Replica fb seed 1).
#
#   bash 0820_queue_stage4a.sh            # empty GPUs (<2000MB) only
#   GPUS="0 1" bash 0820_queue_stage4a.sh
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8"
EL="--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8"
R8REP="--nviews 8 --data-mode r8 --epochs 40 --batch-size 3 --accum 11 --subset-aug --vdrop-kmax 4"
R6MP="--nviews 6 --data-mode r6 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"
R8MP="--nviews 8 --data-mode r8 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"
FBREP="--nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"

JOBS=(
# --- E: stage-3r leftovers (cheap, they settle two interpretations)
"0820_eatllrd_cs_fb_rep_s1|data_0422|$EL --afm-stem conv $FBREP --seed 1"
"0820_sslamllrd_cs_fb_rep_s1|data_0422|$SL --afm-stem conv $FBREP --seed 1"

# --- B early probe: gated mic-differential residual on the A0 stem, MP3D 6/8ch
"0820_sslcs_B2_r6_mp3d|data_mp3d|$SL --afm-stem conv --mic-diff gate $R6MP"
"0820_sslcs_B2_r8_mp3d|data_mp3d|$SL --afm-stem conv --mic-diff gate $R8MP"

# --- A: stem search on the three hard cells (A0 already exists for all three)
"0820_sslcs_A1_r8_mp3d|data_mp3d|$SL --afm-stem conv_res  $R8MP"
"0820_sslcs_A2_r8_mp3d|data_mp3d|$SL --afm-stem conv_ms   $R8MP"
"0820_sslcs_A3_r8_mp3d|data_mp3d|$SL --afm-stem conv_fact $R8MP"
"0820_sslcs_A1_r6_mp3d|data_mp3d|$SL --afm-stem conv_res  $R6MP"
"0820_sslcs_A2_r6_mp3d|data_mp3d|$SL --afm-stem conv_ms   $R6MP"
"0820_sslcs_A3_r6_mp3d|data_mp3d|$SL --afm-stem conv_fact $R6MP"
"0820_sslcs_A1_r8_rep|data_0422|$SL --afm-stem conv_res  $R8REP"
"0820_sslcs_A2_r8_rep|data_0422|$SL --afm-stem conv_ms   $R8REP"
"0820_sslcs_A3_r8_rep|data_0422|$SL --afm-stem conv_fact $R8REP"
)

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for g in $GPUS; do
    [ $i -ge ${#JOBS[@]} ] && break
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$mem" -lt 2000 ]; then
      IFS='|' read -r name dm extra <<< "${JOBS[$i]}"
      if [ -e "comparison_0820/logs/$name.log" ]; then i=$((i+1)); continue; fi
      echo "[dispatch] $name -> GPU $g ($(date +%m/%d\ %H:%M))"
      CUDA_VISIBLE_DEVICES=$g DATA_MODULE=$dm setsid nohup python3 0820_train_oaa_afm.py \
        --run-name $name $extra --lr 5e-4 --out-dir comparison_0820 \
        > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      i=$((i+1)); sleep 90
    fi
  done
  sleep 180
done
echo "[dispatch] all ${#JOBS[@]} stage-4a jobs launched"
