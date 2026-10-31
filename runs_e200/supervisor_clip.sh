#!/usr/bin/env bash
# Detached supervisor for the real-CLIP grid (18 cells). Independent of any chat session or laptop. Idempotent, flock-guarded.
set -u
cd /home/user/Downloads/ML-project/ML-Project
exec 9>runs_e200/logs/supervisor_clip.lock
flock -n 9 || { echo "[SUP] another supervisor holds the lock; exiting"; exit 0; }
echo "[SUP] started $(date -Is) pid=$$"
DEADLINE=$(( $(date +%s) + 12*3600 ))
# queue behind any other training on this shared GPU: 3 consecutive clear checks, 40 s apart
clear=0
while [ $clear -lt 3 ]; do
  if pgrep -f 'launch_e200.py|worker_(cdval|l2cd|cd|native)\.sh' >/dev/null; then clear=0; else clear=$((clear+1)); fi
  [ $(date +%s) -gt $DEADLINE ] && { echo "[SUP] deadline while queued"; exit 1; }
  sleep 40
done
echo "[SUP] GPU clear $(date -Is)"
missing() { local n=0; for T in l1 l2clip l2pclip; do for A in baseline invariant; do for S in 1 2 3; do
  D=runs_e200/mosi_${T}_${A}_seed${S}_cdval; [ -f $D/results.json ] && [ -f $D/checkpoint.pt ] || n=$((n+1)); done; done; done; echo $n; }
for PASS in 1 2 3; do
  M=$(missing); echo "[SUP] pass $PASS: $M of 18 missing $(date -Is)"; [ "$M" -eq 0 ] && break
  [ $(date +%s) -gt $DEADLINE ] && { echo "[SUP] deadline"; break; }
  bash runs_e200/worker_clip.sh baseline & P1=$!
  bash runs_e200/worker_clip.sh invariant & P2=$!
  wait $P1 $P2
done
echo "[SUP] inventory $(date -Is)"
for T in l1 l2clip l2pclip; do for A in baseline invariant; do for S in 1 2 3; do D=runs_e200/mosi_${T}_${A}_seed${S}_cdval
  printf '[SUP] %s json=%s ckpt=%s\n' ${D##*/} $([ -f $D/results.json ] && echo Y || echo N) $([ -f $D/checkpoint.pt ] && echo Y || echo N); done; done; done
if [ "$(missing)" -eq 0 ]; then
  echo "[SUP] fidelity audit $(date -Is)"
  .venv/bin/python tools/fidelity_audit.py --runs_root runs_e200 --device cuda \
    --pattern '^mosi_(l1|l2clip|l2pclip)_(baseline|invariant)_seed[0-9]+_cdval$' \
    --per_run_dir results/fidelity_audit_clip_runs --out results/fidelity_audit_clip.json > results/fidelity_audit_clip.log 2>&1
  echo "[SUP] audit rc=$? $(date -Is)"
  crontab -l 2>/dev/null | grep -v '# clipgrid' | crontab -
  touch runs_e200/logs/supervisor_clip.finished
fi
echo "[SUP] exit $(date -Is)"
