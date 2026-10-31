cd /home/user/Downloads/ML-project/ML-Project
set -o pipefail
START=$(date +%s)
PYTHONUNBUFFERED=1 .venv/bin/python tools/extract_clip_features.py --video_dir /home/user/Downloads/ML-project/CMU-MOSI-raw/Raw \
  --reference_pkl data/mosi_data.pkl --output_path data/mosi_clip_b32_v50_vision.pkl --native_out data/mosi_clip_b32_native.pkl \
  --model ViT-B/32 --seq_len 50 --fps 30 --batch_size 64 --device cuda:0 2>&1 | grep -v -E 'FutureWarning|warnings.warn'
RC=$?
echo "rc=$RC wall_s=$(( $(date +%s) - START ))"
sha256sum data/mosi_clip_b32_v50_vision.pkl data/mosi_clip_b32_native.pkl
echo EXTRACT_DONE
