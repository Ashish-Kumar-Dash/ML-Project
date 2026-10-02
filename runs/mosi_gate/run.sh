#!/usr/bin/env bash
Q=$HOME/Downloads/ML-project/ML-Project; R=$Q/runs/mosi_gate
cd $R/MPLMM
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 15 > $R/gpu.csv 2>&1 & GP=$!
echo "start_epoch=$(date +%s) start_iso=$(date -Iseconds)" > $R/walltime.txt
GATE_OUT=$R GATE_DRAWS=20 /usr/bin/time -v -o $R/time_v.txt $Q/.venv/bin/python -u main.py \
  --pretrained_model $Q/pretrained/mosei.pt --dataset mosi --data_path $Q/data/mosi_data.pkl \
  --drop_rate 0.7 --seed 1 --name $R/mosi_seed1.pt > $R/train.log 2>&1
rc=$?
echo "end_epoch=$(date +%s) end_iso=$(date -Iseconds) exit=$rc" >> $R/walltime.txt
kill $GP 2>/dev/null; touch $R/.done
