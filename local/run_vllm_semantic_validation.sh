#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:18000/v1}"
MODEL_NAME="${MODEL_NAME:-/cache/ma-user/curious_vla_assets/models/Curious-VLA}"
WARMUP_ROOT="${WARMUP_ROOT:-/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage}"
SCENE_LIMIT="${SCENE_LIMIT:-3}"
OUTPUT_JSON="${OUTPUT_JSON:-/cache/ma-user/curious_vla_assets/logs/vllm_semantic_validation_$(date +%Y%m%d_%H%M%S).json}"
SAVE_RAW_DIR="${SAVE_RAW_DIR:-}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-540}"
SELECTION_MODE="${SELECTION_MODE:-diverse-by-command}"

PYTHON_BIN="${PYTHON_BIN:-python}"

ARGS=(
  --base-url "${BASE_URL}"
  --model-name "${MODEL_NAME}"
  --warmup-root "${WARMUP_ROOT}"
  --scene-limit "${SCENE_LIMIT}"
  --run-text-control
  --run-text-planning-control
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --selection-mode "${SELECTION_MODE}"
  --output-json "${OUTPUT_JSON}"
)

if [[ -n "${SAVE_RAW_DIR}" ]]; then
  ARGS+=(--save-raw-dir "${SAVE_RAW_DIR}")
fi

"${PYTHON_BIN}" /home/ma-user/curious_vla/local/run_vllm_semantic_validation.py "${ARGS[@]}" "$@"
