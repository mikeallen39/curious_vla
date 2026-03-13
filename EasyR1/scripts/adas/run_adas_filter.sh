#!/bin/bash
# ADAS single-round pipeline example.
#
# Processes scorer CSV output, identifies dynamic training samples,
# and outputs a token list for the verl dataloader.
#
# Usage:
#   1. Run parallel inference + NAVSIM scoring externally -> scorer CSVs
#   2. Run this script to filter dynamic samples
#   3. Train with: data.token_filter_file=<path_to_txt>
#
# For multi-round ADAS, repeat steps 1-3 with each new checkpoint.

set -euo pipefail
cd "$(dirname "$0")"

# ===== User Configuration (modify before running) =====
PROJECT_ROOT="/path/to/curious_vla"

INFER_FOLDER="$PROJECT_ROOT/EasyR1/checkpoints/adas/your_adas_exp_name" # the same as exp_name in run_adas_infer.sh

python pipeline.py \
    --infer_folder "$INFER_FOLDER" \
    -p 0.1 \
    --conf 0.1
