#!/bin/bash
# 0820 Stage-4b: fusion-side variants, prioritised by expected gain rather than by grid completeness.
#
# Why this replaces the rest of stage-4a's stem search:
#   The common-mode diagnostic (analysis/diff_stats.py, 0820_RESULTS.md) showed MP3D 8ch has TWO
#   failure modes. ConvStem clears the encoder one (cos 1.000 -> 0.778) but not the MAE:
#   sslam+LLRD+cs scores 0.9723 there with an encoder statistically indistinguishable from the only
#   MP3D 8ch success. The cell is a FUSION failure. Consistent with that, the three stem variants
#   A1/A2/A3 all sat at or above A0 on that cell (1.1782 / 1.2034 / -- vs A0's 1.1002), while the
#   gated mic-differential B2 reached 0.8346 by epoch 9. So: stop spending GPU on the stem axis,
#   spend it on the fusion axis. A0 (conv) stays the stem everywhere -- it is 8 wins / 5 ties /
#   1 loss over the linear patch embed in the paired ablation.
#
#   Variants (all zero-init: every model starts bit-identical to A0, verified max|dy| = 0):
#     B1  --mic-diff res        Z_i = S_i + alpha * P(D_i), alpha learnable      control: is a gate needed?
#     B2  --mic-diff gate       channel-wise gated differential                  (already running, MP3D r6/r8)
#     B3  --mic-diff gate_ctx   B2 + mean_j S_j to the decoder ONCE as FiLM
#     C1  --fine-res            Z_i = S_i + g_L * P_L(pooled fine tokens)        isolates the fine path
#     C3  --mic-diff gate --fine-res    Model 3, the recommended final structure
#
#   Cells, in dispatch order: the loss cell (MP3D r8) gets the full ladder first, then Model 3 is
#   carried to the other cells to check it does not cost anything where the cell is already healthy.
#
#   bash 0820_queue_stage4b.sh            # empty GPUs (<2000MB) only
#   GPUS="0 1" bash 0820_queue_stage4b.sh
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv"
R8MP="--nviews 8 --data-mode r8 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"
R6MP="--nviews 6 --data-mode r6 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"
FBMP="--nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"
R8REP="--nviews 8 --data-mode r8 --epochs 40 --batch-size 3 --accum 11 --subset-aug --vdrop-kmax 4"
R2REP="--nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2"

JOBS=(
# --- the loss cell first: full B/C ladder on MP3D r8
"0820_sslcs_C3_r8_mp3d|data_mp3d|$SL --mic-diff gate --fine-res $R8MP"
"0820_sslcs_B3_r8_mp3d|data_mp3d|$SL --mic-diff gate_ctx $R8MP"
"0820_sslcs_C1_r8_mp3d|data_mp3d|$SL --fine-res $R8MP"
"0820_sslcs_B1_r8_mp3d|data_mp3d|$SL --mic-diff res $R8MP"
# --- Model 3 carried to the remaining cells
"0820_sslcs_C3_r6_mp3d|data_mp3d|$SL --mic-diff gate --fine-res $R6MP"
"0820_sslcs_C3_fb_mp3d|data_mp3d|$SL --mic-diff gate --fine-res $FBMP"
"0820_sslcs_C3_r8_rep|data_0422|$SL --mic-diff gate --fine-res $R8REP"
"0820_sslcs_C3_r2_rep|data_0422|$SL --mic-diff gate --fine-res $R2REP"
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
echo "[dispatch] all ${#JOBS[@]} stage-4b jobs launched"
