#!/usr/bin/env bash
# usage: worker_clip.sh ARM -- real-CLIP grid for one arm: L1 (Facet), L2-CLIP (512-d), L2'-CLIP (PLS 20-d) x seeds 1..3,
# 200 epochs, when=30, complete-data validation (--valid_drop_rate 0.0). Skips any cell with results.json + checkpoint.pt.
ARM=$1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /home/user/Downloads/ML-project/ML-Project
for CELL in l1:1 l2clip:1 l1:2 l2clip:2 l1:3 l2clip:3 l2pclip:1 l2pclip:2 l2pclip:3; do
  TAG=${CELL%%:*}; SEED=${CELL##*:}
  case $TAG in l1) F=data/mosi_data.pkl;; l2clip) F=data/clip/mosi_l2clip.pkl;; l2pclip) F=data/clip/mosi_l2pclip.pkl;; esac
  NAME=mosi_${TAG}_${ARM}_seed${SEED}_cdval; RD=runs_e200/$NAME
  if [ -f $RD/results.json ] && [ -f $RD/checkpoint.pt ]; then echo "[SKIP] $NAME"; continue; fi
  echo "[START] $NAME $(date -Is)"
  .venv/bin/python runs_e200/launch_e200.py --variant $ARM --dataset mosi --data_path $F --pretrained_model pretrained/mosei.pt \
    --drop_rate 0.7 --valid_drop_rate 0.0 --batch_size 16 --accum_steps 4 --lr 0.001 --num_epochs 200 --seed $SEED --eval_fidelity --name $RD \
    > runs_e200/logs/$NAME.log 2>&1
  RC=$?; echo "[END] $NAME rc=$RC $(date -Is)"
  [ $RC -eq 0 ] && touch runs_e200/logs/$NAME.done || touch runs_e200/logs/$NAME.failed
done
echo "[WORKER] $ARM finished $(date -Is)"
