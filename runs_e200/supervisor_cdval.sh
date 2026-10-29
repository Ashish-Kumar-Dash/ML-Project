#!/usr/bin/env bash
# Detached supervisor for the L2/L2' complete-data-validation grid (12 cells).
# Independent of any Claude session or the operator's laptop. Idempotent.
set -u
cd /home/user/Downloads/ML-project/ML-Project
LOCK=runs_e200/logs/supervisor_cdval.lock
exec 9>$LOCK
flock -n 9 || { echo "[SUP] another supervisor holds the lock; exiting"; exit 0; }
echo "[SUP] started $(date -Is) pid=$$"

missing() {
  local n=0
  for TAG in l2 l2prime; do for ARM in baseline invariant; do for S in 1 2 3; do
    D=runs_e200/mosi_${TAG}_${ARM}_seed${S}_cdval
    [ -f $D/results.json ] && [ -f $D/checkpoint.pt ] || n=$((n+1))
  done; done; done
  echo $n
}

for PASS in 1 2 3; do
  M=$(missing)
  echo "[SUP] pass $PASS start: $M of 12 cells missing $(date -Is)"
  [ "$M" -eq 0 ] && break
  bash runs_e200/worker_cdval.sh baseline &  P1=$!
  bash runs_e200/worker_cdval.sh invariant & P2=$!
  wait $P1 $P2
  echo "[SUP] pass $PASS done: $(missing) of 12 still missing $(date -Is)"
done

echo "[SUP] final inventory $(date -Is)"
for TAG in l2 l2prime; do for ARM in baseline invariant; do for S in 1 2 3; do
  D=runs_e200/mosi_${TAG}_${ARM}_seed${S}_cdval
  printf '[SUP] %s json=%s ckpt=%s\n' ${D##*/} \
    $([ -f $D/results.json ] && echo Y || echo N) $([ -f $D/checkpoint.pt ] && echo Y || echo N)
done; done; done
touch runs_e200/logs/supervisor_cdval.finished
echo "[SUP] exit $(date -Is)"
