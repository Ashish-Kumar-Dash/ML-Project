#!/usr/bin/env bash
set -euo pipefail
P=$HOME/Downloads/ML-project
echo "[$(date)] start"
python3 -m venv $P/.venv
$P/.venv/bin/pip install --upgrade pip
$P/.venv/bin/pip install torch h5py numpy scikit-learn matplotlib
$P/.venv/bin/pip freeze > $P/requirements.lock.txt
echo "[$(date)] DONE"; touch $P/.venv/.setup_done
