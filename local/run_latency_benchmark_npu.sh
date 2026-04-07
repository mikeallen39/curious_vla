#!/usr/bin/env bash
set -euo pipefail

source /home/ma-user/curious_vla/local/local_env_npu.sh
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$LATENCY_ENV_PREFIX"

mkdir -p "$LATENCY_LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
output_json="$LATENCY_LOG_DIR/latency_benchmark_npu_${timestamp}.json"

python /home/ma-user/curious_vla/local/run_latency_benchmark_npu.py \
  --model-dir "$CURIOUS_VLA_MODEL_DIR" \
  --warmup-root "$WARMUP_ROOT" \
  --output-json "$output_json" \
  "$@"
