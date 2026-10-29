#!/usr/bin/env bash
# Detached supervisor: fidelity audit over the L2 / L2' rungs and the L2 arms C/D,
# QUEUED behind the L2 arms C/D training so it runs on an idle GPU (measured: ~19 s per
# run idle vs ~95 s under contention). Independent of any Claude session or laptop.
# Idempotent: tools/fidelity_audit.py reproduces every stored value before scoring and
# resumes from per-run JSONs, so a re-run is cheap and a partial run is recoverable.
set -u
cd /home/user/Downloads/ML-project/ML-Project
exec 9>runs_e200/logs/supervisor_audit.lock
flock -n 9 || { echo "[AUD] another audit supervisor holds the lock; exiting $(date -Is)"; exit 0; }
echo "[AUD] started $(date -Is) pid=$$ commit=$(git log --oneline -1 | cut -c1-7)"

PAT='^mosi_l2(prime)?_(baseline|invariant|frozen_dense|pool_fixed)_seed[0-9]+(_cdval)?$'

# 1) wait for an idle GPU: no training process of any kind alive, 3 consecutive clear checks
W=0; IDLE=0
while [ $W -lt 43200 ]; do
  if ! pgrep -f 'launch_e200.py|worker_l2cd.sh|worker_cdval.sh|runs_e200_native_v2/worker.sh' >/dev/null; then
    IDLE=$((IDLE+1)); [ $IDLE -ge 3 ] && break
  else
    IDLE=0
  fi
  [ $((W % 1800)) -eq 0 ] && echo "[AUD] waiting for idle GPU ($((W/60)) min) $(date -Is)"
  sleep 40; W=$((W+40))
done
echo "[AUD] GPU idle after $((W/60)) min; starting audit $(date -Is)"

# 2) score every L2 / L2' run (both validation protocols, all four arms)
START=$(date +%s)
.venv/bin/python tools/fidelity_audit.py \
  --runs_root runs_e200 --device cuda \
  --pattern "$PAT" \
  --per_run_dir results/fidelity_audit_l2_runs \
  --out results/fidelity_audit_l2.json >> runs_e200/logs/fidelity_audit_l2.log 2>&1
RC=$?
echo "[AUD] fidelity_audit (L2 family) rc=$RC in $((($(date +%s)-START)/60)) min $(date -Is)"

# 3) report what landed
if [ -f results/fidelity_audit_l2.json ]; then
  .venv/bin/python - <<'PY' >> runs_e200/logs/fidelity_audit_l2.log 2>&1
import json
d = json.load(open("results/fidelity_audit_l2.json"))
runs = d.get("runs", d)
print("[AUD] scored runs:", len(runs) if hasattr(runs, "__len__") else "?")
print("[AUD] skipped:", d.get("skipped_dirs"))
PY
  tail -4 runs_e200/logs/fidelity_audit_l2.log
fi
( crontab -l 2>/dev/null | grep -v '# l2audit' ) | crontab -
touch runs_e200/logs/supervisor_audit.finished
echo "[AUD] exit $(date -Is)"
