#!/usr/bin/env bash
# usage: worker_clip3.sh ARM -- L2'-CLIP-PCA label-free control (data/clip/mosi_l2pcaclip.pkl) x seeds 1..3, same protocol as worker_clip.sh
ARM=$1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /home/user/Downloads/ML-project/ML-Project
for SEED in 1 2 3; do
  NAME=mosi_l2pcaclip_${ARM}_seed${SEED}_cdval; RD=runs_e200/$NAME
  if [ -f $RD/results.json ] && [ -f $RD/checkpoint.pt ]; then echo "[SKIP] $NAME"; continue; fi
  echo "[START] $NAME $(date -Is)"
  .venv/bin/python runs_e200/launch_e200.py --variant $ARM --dataset mosi --data_path data/clip/mosi_l2pcaclip.pkl --pretrained_model pretrained/mosei.pt \
    --drop_rate 0.7 --valid_drop_rate 0.0 --batch_size 16 --accum_steps 4 --lr 0.001 --num_epochs 200 --seed $SEED --eval_fidelity --name $RD \
    > runs_e200/logs/$NAME.log 2>&1
  RC=$?; echo "[END] $NAME rc=$RC $(date -Is)"
  [ $RC -eq 0 ] && touch runs_e200/logs/$NAME.done || touch runs_e200/logs/$NAME.failed
done
echo "[WORKER] $ARM finished $(date -Is)"
