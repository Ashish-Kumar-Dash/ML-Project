#!/usr/bin/env bash
# usage: worker6.sh SEED VARIANT -- runs l50,l100,l200 for one (seed, variant); 200 epochs, when=30
SEED=$1; V=$2
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd ~/Downloads/ML-project/ML-Project
for L in 50 100 200; do
  NAME=mosi_ladder_l${L}_${V}_seed${SEED}
  RD=runs_e200/$NAME
  if [ -f $RD/results.json ] && [ -f $RD/checkpoint.pt ]; then echo "[SKIP] $NAME"; continue; fi
  echo "[START] $NAME $(date -Is)"
  .venv/bin/python runs_e200/launch_e200.py --variant $V --dataset mosi \
    --data_path data/ladder/mosi_data_noalign_l${L}.pkl --pretrained_model pretrained/mosei.pt \
    --drop_rate 0.7 --batch_size 16 --accum_steps 4 --lr 0.001 --num_epochs 200 --seed $SEED \
    --eval_fidelity --name $RD > runs_e200/logs/$NAME.log 2>&1
  RC=$?
  echo "[END] $NAME rc=$RC $(date -Is)"
  [ $RC -eq 0 ] && touch runs_e200/logs/$NAME.done || touch runs_e200/logs/$NAME.failed
done
touch runs_e200/logs/worker_seed${SEED}_${V}.done
