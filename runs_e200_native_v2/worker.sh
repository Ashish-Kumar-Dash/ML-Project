#!/usr/bin/env bash
# Native rung (A=375, V=500) rerun, 200 epochs, when=30 -- one arm, seeds 2,3 then 1.
# Seeds 2/3 complete n=3; seed 1 is a byte-identity check against runs_e200/ (same machine).
# Same command as runs_e200/worker_native.sh except the output root. Up to 3 attempts per seed.
V=$1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /home/user/Downloads/ML-project/ML-Project
for SEED in 2 3 1; do
  NAME=mosi_ladder_native_${V}_seed${SEED}; RD=runs_e200_native_v2/$NAME
  for TRY in 1 2 3; do
    if [ -f $RD/results.json ] && [ -f $RD/checkpoint.pt ] && grep -q '"launcher"' $RD/results.json; then echo "[DONE] $NAME"; break; fi
    echo "[START] $NAME try=$TRY $(date -Is)"
    .venv/bin/python runs_e200/launch_e200.py --variant $V --dataset mosi \
      --data_path data/ladder/mosi_data_noalign_native.pkl --pretrained_model pretrained/mosei.pt \
      --drop_rate 0.7 --batch_size 16 --accum_steps 4 --lr 0.001 --num_epochs 200 --seed $SEED \
      --eval_fidelity --name $RD > runs_e200_native_v2/logs/${NAME}_try${TRY}.log 2>&1
    RC=$?; echo "[END] $NAME try=$TRY rc=$RC $(date -Is)"
    [ $RC -eq 0 ] && touch runs_e200_native_v2/logs/$NAME.done && break
    touch runs_e200_native_v2/logs/${NAME}_try${TRY}.failed; sleep 60
  done
done
touch runs_e200_native_v2/logs/worker_${V}.finished
