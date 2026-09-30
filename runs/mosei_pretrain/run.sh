#!/usr/bin/env bash
Q=$HOME/Downloads/ML-project/ML-Project; R=$Q/runs/mosei_pretrain
cd $R/MPLMM
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 > $R/gpu.csv 2>&1 & GP=$!
echo "start_epoch=$(date +%s) start_iso=$(date -Iseconds)" > $R/walltime.txt
/usr/bin/time -v -o $R/time_v.txt $Q/.venv/bin/python -u main.py --dataset mosei \
  --data_path $Q/data/mosei_senti_data.pkl --drop_rate 0 --batch_size 64 --name $Q/pretrained/mosei.pt
rc=$?
echo "end_epoch=$(date +%s) end_iso=$(date -Iseconds) exit=$rc" >> $R/walltime.txt
kill $GP 2>/dev/null; touch $R/.done
