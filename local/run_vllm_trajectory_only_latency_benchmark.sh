#!/usr/bin/env bash
set -euo pipefail

source /home/ma-user/curious_vla/local/local_env_vllm_ascend.sh
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$VLLM_ASCEND_ENV_PREFIX"

mkdir -p "$LATENCY_LOG_DIR"

BASE_URL="${BASE_URL:-$VLLM_ASCEND_BASE_URL}"
SCENE_LIMIT="${SCENE_LIMIT:-4}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"
BENCHMARK_RUNS="${BENCHMARK_RUNS:-50}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-704}"
SELECTION_MODE="${SELECTION_MODE:-diverse-by-command}"
MAX_TOKENS="${MAX_TOKENS:-256}"

timestamp="$(date +%Y%m%d_%H%M%S)"
output_json="$LATENCY_LOG_DIR/vllm_trajectory_only_latency_benchmark_${timestamp}.json"

python /home/ma-user/curious_vla/local/run_vllm_trajectory_only_latency_benchmark.py \
  --base-url "$BASE_URL" \
  --model-name "$CURIOUS_VLA_MODEL_DIR" \
  --warmup-root "$WARMUP_ROOT" \
  --scene-limit "$SCENE_LIMIT" \
  --warmup-runs "$WARMUP_RUNS" \
  --benchmark-runs "$BENCHMARK_RUNS" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --selection-mode "$SELECTION_MODE" \
  --max-tokens "$MAX_TOKENS" \
  --output-json "$output_json" \
  "$@"
