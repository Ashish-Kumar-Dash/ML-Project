#!/usr/bin/env bash
# usage: worker_cd.sh VARIANT   -- runs l200 x {frozen_dense|pool_fixed} seeds 1..3, 200 epochs, when=30
V=$1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /home/user/Downloads/ML-project/ML-Project
for SEED in 1 2 3; do
  NAME=mosi_ladder_l200_${V}_seed${SEED}
  RD=runs_e200/$NAME
  if [ -f $RD/results.json ] && [ -f $RD/checkpoint.pt ]; then echo "[SKIP] $NAME"; continue; fi
  echo "[START] $NAME $(date -Is)"
  .venv/bin/python runs_e200/launch_e200.py --variant $V --dataset mosi \
    --data_path data/ladder/mosi_data_noalign_l200.pkl --pretrained_model pretrained/mosei.pt \
    --drop_rate 0.7 --batch_size 16 --accum_steps 4 --lr 0.001 --num_epochs 200 --seed $SEED \
    --eval_fidelity --name $RD > runs_e200/logs/$NAME.log 2>&1
  RC=$?
  echo "[END] $NAME rc=$RC $(date -Is)"
  [ $RC -eq 0 ] && touch runs_e200/logs/$NAME.done || touch runs_e200/logs/$NAME.failed
done
touch runs_e200/logs/worker_cd_${V}.done
