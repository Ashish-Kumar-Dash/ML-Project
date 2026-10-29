#!/usr/bin/env bash
# Detached supervisor: arms C and D on the L2 rung, QUEUED behind the native_v2 rung so the
# two never share the GPU (this box saturates its SMs with two runs; a third only slows
# everything). Independent of any Claude session and of the operator's laptop. Idempotent
# (flock + skip-if-done). Restarted by an @reboot cron line tagged '# l2cd', which this
# script removes once all six cells are complete.
set -u
cd /home/user/Downloads/ML-project/ML-Project
mkdir -p runs_e200/logs
exec 9>runs_e200/logs/supervisor_l2cd.lock
flock -n 9 || { echo "[SUP] another supervisor holds the lock; exiting $(date -Is)"; exit 0; }
echo "[SUP] started $(date -Is) pid=$$ commit=$(git log --oneline -1 | cut -c1-7)"

complete_count() {
  local n=0
  for ARM in frozen_dense pool_fixed; do for S in 1 2 3; do
    D=runs_e200/mosi_l2_${ARM}_seed${S}_cdval
    [ -f $D/results.json ] && [ -f $D/checkpoint.pt ] && n=$((n+1))
  done; done
  echo $n
}

# 1) Queue behind the native rung: wait while its supervisor OR any training process is alive.
#    Cap at 12 h so a stuck upstream job cannot park this forever.
W=0
while [ $W -lt 43200 ]; do
  if ! pgrep -f "runs_e200_native_v2/supervisor.sh" >/dev/null && ! pgrep -f "launch_e200.py" >/dev/null; then
    IDLE=$((IDLE+1)) 2>/dev/null || IDLE=1
    [ "${IDLE:-0}" -ge 3 ] && break      # 3 consecutive clear checks (~2 min) before starting
  else
    IDLE=0
  fi
  [ $((W % 1800)) -eq 0 ] && echo "[SUP] waiting for GPU ($((W/60)) min elapsed) $(date -Is)"
  sleep 40; W=$((W+40))
done
echo "[SUP] GPU clear after $((W/60)) min; starting arms C/D at L2 $(date -Is)"

# 2) Both arms in parallel, three seeds each, up to 3 passes
for PASS in 1 2 3; do
  M=$(complete_count)
  echo "[SUP] pass $PASS start: $M of 6 cells complete $(date -Is)"
  [ "$M" -eq 6 ] && break
  bash runs_e200/worker_l2cd.sh frozen_dense >> runs_e200/logs/worker_l2cd_frozen_dense.log 2>&1 & P1=$!
  bash runs_e200/worker_l2cd.sh pool_fixed   >> runs_e200/logs/worker_l2cd_pool_fixed.log 2>&1 & P2=$!
  wait $P1; wait $P2
  echo "[SUP] pass $PASS done: $(complete_count) of 6 complete $(date -Is)"
done

echo "[SUP] final inventory $(date -Is)"
for ARM in frozen_dense pool_fixed; do for S in 1 2 3; do
  D=runs_e200/mosi_l2_${ARM}_seed${S}_cdval
  printf '[SUP] %s json=%s ckpt=%s\n' ${D##*/} \
    $([ -f $D/results.json ] && echo Y || echo N) $([ -f $D/checkpoint.pt ] && echo Y || echo N)
done; done
if [ "$(complete_count)" -eq 6 ]; then
  ( crontab -l 2>/dev/null | grep -v '# l2cd' ) | crontab -
  echo "[SUP] removed @reboot cron line"
fi
touch runs_e200/logs/supervisor_l2cd.finished
echo "[SUP] exit $(date -Is)"
