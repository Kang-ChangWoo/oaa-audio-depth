#!/bin/bash
# 0820 Stage-5: the INPUT-side study. Two independent defects were found in how the echo reaches
# the pretrained backbone; this queue tests a fix for each, plus the combination.
#
# Defect 1 -- the pretrained input interface is thrown away.
#   The AFM checkpoints ship a pretrained patch embedding (sslam/eat: local_encoder.proj, a
#   768x1x16x16 conv trained on AudioSet-2M). The released code discards it, trains a fresh
#   task-specific patch embed, and bicubic-interpolates the positional embedding from the native
#   (64 time, 8 freq) grid onto a (16 freq, 32 time) grid -- i.e. every layer that actually touches
#   the data starts from scratch, and the axis meaning of the position code is resampled.
#   FIX: --afm-stem native. The spectrogram is mel-compressed (128 bins, 48 kHz filterbank) and
#   re-laid-out as the backbone's own 1024x128 (time, mel) image, so the pretrained patch embed AND
#   the positional embedding transfer VERBATIM. The native grid holds 64x8 = 512 tokens, exactly the
#   count OAA needs, so nothing downstream changes. Transfer goes 97.3% -> 99.8% of AFM params, and
#   the checkpoint reports 0 unused source tensors for the first time.
#   Legal because RayMicAttn's geometric bias is per-OBSERVATION, not per-token
#   (bmic.unsqueeze(2).expand(R,N,M,h)) -- the encoder token layout carries no geometry.
#
# Defect 2 -- the time axis, which encodes distance, carries almost no information.
#   The true STFT of the 58.8 ms clip at hop 160 is 257 freq x 18 time. The cache then
#   nearest-upsamples that to 256x512, replicating the time axis 28x. At the 16x16 patch grid each
#   patch column spans 0.56 real frames, so most patches contain no temporal variation at all --
#   there is nothing for a pretrained temporal model to use. Frequency is compressed 16x while time
#   is expanded 1.8x, which is backwards for a task where time IS distance.
#   FIX: STFT_HOP=44 (0.92 ms) -> exactly 64 real frames, one per native time token. N_FFT and the
#   window are unchanged, so the spectral content is identical; only the sampling of the time axis
#   changes. This is a PREPROCESSING change, so the CNN baseline is retrained on it too -- otherwise
#   the comparison would price the input recipe, not the encoder.
#
# Controls already in hand at hop 160: sslam+LLRD+cs Rep fb 0.2560 / MP3D fb 0.7744;
# OAA-CNN (paper) Rep fb 0.2596 / MP3D fb 0.7849. Test 1 supplies native @ hop 160.
#
#   bash 0820_queue_stage5_input.sh            # empty GPUs (<2000MB) only
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8"
CN="--audio-backbone cnn --warmup-ep 4"
R2REP="--nviews 2 --data-mode r2 --epochs 40 --batch-size 12 --accum 2"
FBREP="--nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"
FBMP="--nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"
R8MP="--nviews 8 --data-mode r8 --epochs 30 --batch-size 4 --accum 1 --stem-stride1"

#  name | data module | STFT hop | trainer args
JOBS=(
# --- Test 2 first: hop 44 puts real information on the distance axis, and it is the arm where the
#     native layout can actually pay off (64 real frames = 64 native time tokens, 1:1).
#     CNN control leads: if the CNN gains as much, the win is the input recipe, not the encoder.
"0820_h44_cnn_fb_rep|data_0422|44|$CN $FBREP"
"0820_h44_cnn_fb_mp3d|data_mp3d|44|$CN $FBMP"
"0820_h44_sslcs_fb_rep|data_0422|44|$SL --afm-stem conv $FBREP"
"0820_h44_sslcs_fb_mp3d|data_mp3d|44|$SL --afm-stem conv $FBMP"
# --- Test 1 + 2 together: the configuration both defects argue for
"0820_h44_sslnat_fb_rep|data_0422|44|$SL --afm-stem native $FBREP"
"0820_h44_sslnat_fb_mp3d|data_mp3d|44|$SL --afm-stem native $FBMP"
# --- Test 1 alone, at the released hop. This is the WEAKEST cell for the native layout: at hop 160
#     only ~18 real frames exist, so 64 native time tokens are 3.5x oversampled where the current 32
#     are 1.8x -- the layout fixes the transfer while making the time-axis waste worse. It is carried
#     only as the hop-160 corner of the 2x2 (native on/off x hop 44/160), which is what separates
#     "the pretrained interface transfers" from "the time axis was empty". The two unpaired
#     native@160 cells (Rep r2, MP3D r8) were dropped: with no hop-44 partner a result there could
#     not be attributed to either factor.
"0820_sslnat_fb_rep|data_0422|160|$SL --afm-stem native $FBREP"
"0820_sslnat_fb_mp3d|data_mp3d|160|$SL --afm-stem native $FBMP"
)

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for g in $GPUS; do
    [ $i -ge ${#JOBS[@]} ] && break
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$mem" -lt 2000 ]; then
      IFS='|' read -r name dm hop extra <<< "${JOBS[$i]}"
      if [ -e "comparison_0820/logs/$name.log" ]; then i=$((i+1)); continue; fi
      echo "[dispatch] $name (hop $hop) -> GPU $g ($(date +%m/%d\ %H:%M))"
      CUDA_VISIBLE_DEVICES=$g DATA_MODULE=$dm STFT_HOP=$hop setsid nohup python3 0820_train_oaa_afm.py \
        --run-name $name $extra --lr 5e-4 --out-dir comparison_0820 \
        > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      i=$((i+1)); sleep 90
    fi
  done
  sleep 180
done
echo "[dispatch] all ${#JOBS[@]} stage-5 input jobs launched"
