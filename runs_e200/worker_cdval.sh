#!/usr/bin/env bash
# usage: worker_cdval.sh ARM   -- L2 and L2' x seeds 1..3 for one arm, 200 epochs,
# when=30, checkpoint selection on COMPLETE-DATA validation (--valid_drop_rate 0.0).
# Writes to *_cdval dirs; the original mosi_l2*/mosi_l2prime* records are untouched.
ARM=$1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /home/user/Downloads/ML-project/ML-Project
for R in l2:mosi_data_l2.pkl l2prime:mosi_data_l2_prime.pkl; do
  TAG=${R%%:*}; F=${R##*:}
  for SEED in 1 2 3; do
    NAME=mosi_${TAG}_${ARM}_seed${SEED}_cdval
    RD=runs_e200/$NAME
    if [ -f $RD/results.json ] && [ -f $RD/checkpoint.pt ]; then echo "[SKIP] $NAME"; continue; fi
    echo "[START] $NAME $(date -Is)"
    .venv/bin/python runs_e200/launch_e200.py --variant $ARM --dataset mosi \
      --data_path data/$F --pretrained_model pretrained/mosei.pt \
      --drop_rate 0.7 --valid_drop_rate 0.0 --batch_size 16 --accum_steps 4 --lr 0.001 \
      --num_epochs 200 --seed $SEED --eval_fidelity --name $RD \
      > runs_e200/logs/$NAME.log 2>&1
    RC=$?
    echo "[END] $NAME rc=$RC $(date -Is)"
    [ $RC -eq 0 ] && touch runs_e200/logs/$NAME.done || touch runs_e200/logs/$NAME.failed
  done
done
echo "[WORKER] $ARM finished $(date -Is)"
