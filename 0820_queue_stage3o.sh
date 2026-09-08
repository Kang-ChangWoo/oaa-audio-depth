#!/bin/bash
# 0820 Stage-3o: re-queue the two stage-3m jobs (eatllrd_cs_r6_rep / cs_r8_rep) that were
# deferred via placeholder logs to give stage-3n (Beyond-I2D / ITD baseline) queue priority.
# Waits until all 8 beyond jobs have been dispatched, then removes the placeholders and
# dispatches on empty GPUs (<2000MB) only.
cd "$(dirname "$0")"
export AFM_WEIGHTS=/root/local1/changwoo/_afm_weights HF_HOME=/root/local1/changwoo/_afm_weights
export REPLICA_ROOT=/root/local2/replica_0422_lite MP3D_ROOT=/root/local1/changwoo/matterport3d_0303renew R0422_SPLIT=off3
mkdir -p comparison_0820/logs

BEYOND="0820_beyond_fb_rep 0820_beyond_fb_mp3d 0820_beyond_r2_rep 0820_beyond_r2_mp3d 0820_beyond_r6_rep 0820_beyond_r6_mp3d 0820_beyond_r8_rep 0820_beyond_r8_mp3d"
while :; do
  ok=1
  for b in $BEYOND; do [ -e "comparison_0820/logs/$b.log" ] || { ok=0; break; }; done
  [ $ok -eq 1 ] && break
  sleep 300
done
echo "[stage3o] all beyond jobs dispatched -> releasing deferred cs jobs ($(date))"

for n in 0820_eatllrd_cs_r6_rep 0820_eatllrd_cs_r8_rep; do
  grep -q "\[placeholder\]" "comparison_0820/logs/$n.log" 2>/dev/null && rm -f "comparison_0820/logs/$n.log"
done

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
JOBS=(
"0820_eatllrd_cs_r6_rep|data_0422|--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv --nviews 6 --data-mode r6 --epochs 40 --batch-size 4 --accum 8"
"0820_eatllrd_cs_r8_rep|data_0422|--audio-backbone eat --afm-llrd 0.75 --warmup-ep 8 --afm-stem conv --nviews 8 --data-mode r8 --epochs 40 --batch-size 3 --accum 11 --subset-aug --vdrop-kmax 4"
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
echo "[dispatch] all ${#JOBS[@]} stage-3o jobs launched"
