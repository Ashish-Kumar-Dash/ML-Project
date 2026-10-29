#!/usr/bin/env bash
# usage: worker_l2cd.sh ARM   -- L2 rung (wav2vec2/CLIP 768/512, aligned 50/50/50),
# arms C (frozen_dense) and D (pool_fixed), seeds 1..3, 200 epochs, when=30,
# complete-data checkpoint selection (--valid_drop_rate 0.0) to MATCH the *_cdval L2 grid
# these runs are compared against. Up to 3 tries per seed.
ARM=$1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /home/user/Downloads/ML-project/ML-Project
for SEED in 1 2 3; do
  NAME=mosi_l2_${ARM}_seed${SEED}_cdval
  RD=runs_e200/$NAME
  for TRY in 1 2 3; do
    if [ -f $RD/results.json ] && [ -f $RD/checkpoint.pt ]; then echo "[DONE] $NAME"; break; fi
    echo "[START] $NAME try=$TRY $(date -Is)"
    .venv/bin/python runs_e200/launch_e200.py --variant $ARM --dataset mosi \
      --data_path data/mosi_data_l2.pkl --pretrained_model pretrained/mosei.pt \
      --drop_rate 0.7 --valid_drop_rate 0.0 --batch_size 16 --accum_steps 4 --lr 0.001 \
      --num_epochs 200 --seed $SEED --eval_fidelity --name $RD \
      > runs_e200/logs/${NAME}_try${TRY}.log 2>&1
    RC=$?; echo "[END] $NAME try=$TRY rc=$RC $(date -Is)"
    [ $RC -eq 0 ] && touch runs_e200/logs/$NAME.done && break
    touch runs_e200/logs/${NAME}_try${TRY}.failed; sleep 60
  done
done
touch runs_e200/logs/worker_l2cd_${ARM}.finished
echo "[WORKER] $ARM finished $(date -Is)"
