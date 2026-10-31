#!/usr/bin/env bash
set -u
cd /home/user/Downloads/ML-project/ML-Project
exec 9>runs_e200/logs/supervisor_clip2.lock
flock -n 9 || { echo "[SUP] another supervisor holds the lock; exiting"; exit 0; }
echo "[SUP] started $(date -Is) pid=$$"; DEADLINE=$(( $(date +%s) + 6*3600 )); clear=0
while [ $clear -lt 3 ]; do
  if pgrep -f 'launch_e200.py|worker_(cdval|l2cd|cd|native|clip)\.sh' >/dev/null; then clear=0; else clear=$((clear+1)); fi
  [ $(date +%s) -gt $DEADLINE ] && { echo "[SUP] deadline while queued"; exit 1; }; sleep 40
done
missing() { local n=0; for A in baseline invariant; do for S in 1 2 3; do D=runs_e200/mosi_l2pclipsc_${A}_seed${S}_cdval
  [ -f $D/results.json ] && [ -f $D/checkpoint.pt ] || n=$((n+1)); done; done; echo $n; }
for PASS in 1 2 3; do
  M=$(missing); echo "[SUP] pass $PASS: $M of 6 missing $(date -Is)"; [ "$M" -eq 0 ] && break
  bash runs_e200/worker_clip2.sh baseline & P1=$!; bash runs_e200/worker_clip2.sh invariant & P2=$!; wait $P1 $P2
done
if [ "$(missing)" -eq 0 ]; then
  .venv/bin/python tools/fidelity_audit.py --runs_root runs_e200 --device cuda --pattern '^mosi_l2pclipsc_(baseline|invariant)_seed[0-9]+_cdval$' \
    --per_run_dir results/fidelity_audit_clip2_runs --out results/fidelity_audit_clip2.json > results/fidelity_audit_clip2.log 2>&1
  echo "[SUP] audit rc=$? $(date -Is)"
  crontab -l 2>/dev/null | grep -v '# clipgrid2' > /tmp/cron_rest.txt; crontab /tmp/cron_rest.txt; rm -f /tmp/cron_rest.txt
  touch runs_e200/logs/supervisor_clip2.finished
fi
echo "[SUP] exit $(date -Is)"
