#!/usr/bin/env bash
# usage: worker.sh SEED   -- runs l50,l100,l200 x baseline,invariant for one seed, 200 epochs, when=30
SEED=$1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd ~/Downloads/ML-project/ML-Project
for L in 50 100 200; do for V in baseline invariant; do
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
done; done
touch runs_e200/logs/worker_seed${SEED}.done
