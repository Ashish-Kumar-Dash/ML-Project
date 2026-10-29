#!/usr/bin/env bash
# Detached supervisor: native rung rerun (200 epochs) into runs_e200_native_v2/. Independent of any Claude
# session and of the operator's laptop. Idempotent (flock + skip-if-done); restarted by an @reboot
# cron line tagged '# native_v2' which this script removes when everything has finished.
set -u
cd /home/user/Downloads/ML-project/ML-Project
mkdir -p runs_e200_native_v2/logs
exec 9>runs_e200_native_v2/logs/supervisor.lock
flock -n 9 || { echo "[SUP] another supervisor holds the lock; exiting $(date -Is)"; exit 0; }
echo "[SUP] started $(date -Is) pid=$$ commit=$(git log --oneline -1 | cut -c1-7)"
# 1) queue behind the L2/L2' grid: wait while its supervisor is alive and not finished (cap 6 h)
W=0
while pgrep -f "supervisor_cdval.sh" >/dev/null && [ ! -f runs_e200/logs/supervisor_cdval.finished ] && [ $W -lt 21600 ]; do
  [ $((W % 1800)) -eq 0 ] && echo "[SUP] waiting for L2/L2' grid ($((W/60)) min) $(date -Is)"
  sleep 60; W=$((W+60))
done
echo "[SUP] GPU queue clear after $((W/60)) min $(date -Is)"
# 2) both arms in parallel, as in the original native runs
bash runs_e200_native_v2/worker.sh baseline  >> runs_e200_native_v2/logs/worker_baseline.log 2>&1 & P1=$!
bash runs_e200_native_v2/worker.sh invariant >> runs_e200_native_v2/logs/worker_invariant.log 2>&1 & P2=$!
wait $P1; wait $P2
N=0; for A in baseline invariant; do for S in 1 2 3; do D=runs_e200_native_v2/mosi_ladder_native_${A}_seed${S}; [ -f $D/results.json ] && [ -f $D/checkpoint.pt ] && N=$((N+1)); done; done
echo "[SUP] training finished: $N of 6 runs complete $(date -Is)"
# 3) post-steps (eval only)
if [ -f runs_e200_native_v2/mosi_ladder_native_baseline_seed1/results.json ] && [ -f runs_e200_native_v2/mosi_ladder_native_invariant_seed1/results.json ]; then
  .venv/bin/python runs_e200_native_v2/seed1_identity.py >> runs_e200_native_v2/logs/identity.log 2>&1; echo "[SUP] identity check rc=$? $(date -Is)"
fi
.venv/bin/python tools/fidelity_audit.py --runs_root runs_e200_native_v2 --device cuda --per_run_dir results/fidelity_audit_native_v2_runs \
  --out results/fidelity_audit_native_v2.json >> runs_e200_native_v2/logs/fidelity_audit.log 2>&1; echo "[SUP] fidelity audit rc=$? $(date -Is)"
if [ $N -eq 6 ]; then
  ( crontab -l 2>/dev/null | grep -v '# native_v2' ) | crontab - ; echo "[SUP] removed @reboot cron line"
  touch runs_e200_native_v2/logs/ALL_DONE
fi
echo "[SUP] exit $(date -Is)"
