#!/usr/bin/env bash
# Detached supervisor: guarantees l=200 arms C/D seed 3 completes, independent of
# any Claude session or the operator's laptop. Idempotent: if the records already
# exist it runs nothing. Started 2026-10-28.
set -u
REPO=/home/user/Downloads/ML-project/ML-Project
cd $REPO
LOCK=runs_e200/logs/supervisor_cd.lock
exec 9>$LOCK
flock -n 9 || { echo "[SUP] another supervisor holds the lock; exiting"; exit 0; }
echo "[SUP] started $(date -Is) pid=$$"

complete() {  # arm -> 0 if seed 3 record complete
  local d=runs_e200/mosi_ladder_l200_$1_seed3
  [ -f $d/results.json ] && [ -f $d/checkpoint.pt ]
}

DEADLINE=$(( $(date +%s) + 12*3600 ))
while [ $(date +%s) -lt $DEADLINE ]; do
  if complete frozen_dense && complete pool_fixed; then
    echo "[SUP] both seed-3 records complete $(date -Is)"; break
  fi
  if pgrep -f 'launch_e200|worker_cd.sh' > /dev/null; then
    sleep 120; continue
  fi
  echo "[SUP] GPU idle and seed 3 incomplete; running missing arms sequentially $(date -Is)"
  for A in frozen_dense pool_fixed; do
    complete $A && { echo "[SUP] $A seed3 already complete"; continue; }
    echo "[SUP] launching worker_cd.sh $A $(date -Is)"
    bash runs_e200/worker_cd.sh $A
    echo "[SUP] worker_cd.sh $A returned rc=$? $(date -Is)"
  done
done
echo "[SUP] exit $(date -Is)"
for A in frozen_dense pool_fixed; do
  for S in 1 2 3; do
    D=runs_e200/mosi_ladder_l200_${A}_seed${S}
    printf '[SUP] %s json=%s ckpt=%s\n' $D $([ -f $D/results.json ] && echo Y || echo N) $([ -f $D/checkpoint.pt ] && echo Y || echo N)
  done
done
touch runs_e200/logs/supervisor_cd.finished
