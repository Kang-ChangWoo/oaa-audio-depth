#!/bin/bash
# 0820 Stage-9: separate the three things "input resolution" was conflating.
#
# WHY THIS DESIGN DIFFERS FROM A hop x n_fft GRID.
#
# A numerical check of the STFT changes what the hop-44 result can mean:
#
#   window 400 samples = 8.33 ms  -> the time axis can RESOLVE 143 cm of round-trip distance
#   hop 160            = 3.33 ms  -> hop <= win/2 = 200 is already satisfied
#   hop 44             = 0.92 ms  -> 4.5x OVERSAMPLED relative to that window
#
# So hop 44 adds almost no genuinely new information: the resolution ceiling is the WINDOW, and the
# released hop already samples below it. Yet hop 44 is worth up to -0.245 MAE. That points at a
# REPRESENTATION effect (how the 512-wide image is tokenised) rather than an information effect --
# and it means the untested lever is the window, which at 143 cm is coarse for a task whose errors
# are ~25 cm. Three axes therefore have to be separated, not crossed:
#
#   A. WINDOW (what can be resolved).  n_fft stays 512 so the frequency bin count is CONSTANT --
#      a smaller n_fft would cut bins from 257 to 65 and confound frequency resolution into the
#      time answer. Zero-padded short windows keep the two axes independent.
#        win 400 -> 143 cm | win 200 -> 71 cm | win 100 -> 36 cm
#   B. HOP (how densely it is sampled) at fixed window: 160 / 88 / 44 / 22 -> 18 / 32 / 64 / 128 frames.
#   C. TOKEN ALLOCATION -- the replication control, and the cheapest test of the story.
#      The patch grid is 16 freq x 32 time. At hop 160 only 18 distinct frames sit behind those 32
#      time tokens, so the grid over-allocates time. --afm-patch 8,32 re-splits the SAME 512 tokens
#      as 32 freq x 16 time, which matches the real frame count without touching the data at all.
#      If that recovers the hop-44 gain, the bottleneck was token allocation (replication);
#      if it does not, the bottleneck was real information. The mirror control --afm-patch 32,8 at
#      hop 44 (8 freq x 64 time, one token per real frame) tests the same claim from the other side.
#
# Cells: Replica 4ch and MP3D 4ch, the two cells with an existing hop-160 AND hop-44 number
# (Rep 0.2573 -> 0.2496, MP3D 0.7744 -> 0.7562) so every run below has two anchors. Seed 0 first;
# seeds go on whatever separates. Dispatch order puts the token control first -- it is the cheapest
# discriminator and it needs no new preprocessing.
#
#   bash 0820_queue_stage9_input.sh
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

SL="--audio-backbone sslam --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv"
FB="--nviews 4 --data-mode fb --epochs 40 --batch-size 8 --accum 4"

#  name | data module | HOP | WIN | extra trainer args
JOBS=(
# --- C. token-allocation control (no preprocessing change at all)
"0820_s9_tok832_h160_rep|data_0422|160|400|$SL $FB --afm-patch 8,32"
"0820_s9_tok832_h160_mp3d|data_mp3d|160|400|$SL $FB --afm-patch 8,32"
"0820_s9_tok328_h44_rep|data_0422|44|400|$SL $FB --afm-patch 32,8"
"0820_s9_tok328_h44_mp3d|data_mp3d|44|400|$SL $FB --afm-patch 32,8"
# --- A. window at the RELEASED hop: does sharpening time help without denser sampling?
"0820_s9_w100_h160_rep|data_0422|160|100|$SL $FB"
"0820_s9_w100_h160_mp3d|data_mp3d|160|100|$SL $FB"
"0820_s9_w200_h160_mp3d|data_mp3d|160|200|$SL $FB"
# --- A x B. window at hop 44: if the gain were about resolution, a shorter window should add to it
"0820_s9_w100_h44_rep|data_0422|44|100|$SL $FB"
"0820_s9_w100_h44_mp3d|data_mp3d|44|100|$SL $FB"
"0820_s9_w200_h44_mp3d|data_mp3d|44|200|$SL $FB"
# --- B. the rest of the hop sweep at the released window
"0820_s9_h88_mp3d|data_mp3d|88|400|$SL $FB"
"0820_s9_h22_mp3d|data_mp3d|22|400|$SL $FB"
"0820_s9_h88_rep|data_0422|88|400|$SL $FB"
"0820_s9_h22_rep|data_0422|22|400|$SL $FB"
)

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for g in $GPUS; do
    [ $i -ge ${#JOBS[@]} ] && break
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$mem" -lt 2000 ]; then
      IFS='|' read -r name dm hop win extra <<< "${JOBS[$i]}"
      if [ -e "comparison_0820/logs/$name.log" ]; then i=$((i+1)); continue; fi
      echo "[dispatch] $name (hop $hop win $win) -> GPU $g ($(date +%m/%d\ %H:%M))"
      CUDA_VISIBLE_DEVICES=$g DATA_MODULE=$dm STFT_HOP=$hop STFT_WIN=$win setsid nohup \
        python3 0820_train_oaa_afm.py --run-name $name $extra --lr 5e-4 --out-dir comparison_0820 \
        > comparison_0820/logs/$name.log 2>&1 < /dev/null &
      i=$((i+1)); sleep 90
    fi
  done
  sleep 180
done
echo "[dispatch] all ${#JOBS[@]} stage-9 input jobs launched"
