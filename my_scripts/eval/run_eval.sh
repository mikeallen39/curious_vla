#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$LOCAL_ROOT/local_env.sh"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$NAVSIM_ENV_PREFIX"

: "${RUN_SPLIT:=warmup_two_stage}"
: "${AGENT_NAME:=navsim_qwen_norm_cot_baseline_agent}"
: "${EXPERIMENT_NAME:=demo_${RUN_SPLIT}}"
: "${MODEL_NAME_OR_PATH:=$CURIOUS_VLA_MODEL_DIR}"
: "${NAVSIM_LOG_PATH_OVERRIDE:=$TEST_LOG_PATH}"
: "${ORIGINAL_SENSOR_PATH_OVERRIDE:=$ORIGINAL_SENSOR_PATH}"
: "${SYNTHETIC_SENSOR_PATH_OVERRIDE:=$SYNTHETIC_SENSOR_PATH}"
: "${SYNTHETIC_SCENES_PATH_OVERRIDE:=$WARMUP_SYNTHETIC_SCENES_PATH}"
: "${CACHE_PATH_OVERRIDE:=$NAVSIM_EXP_ROOT/metric_cache_${RUN_SPLIT}}"

mkdir -p "$NAVSIM_EXP_ROOT"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py" \
  train_test_split="$RUN_SPLIT" \
  experiment_name="$EXPERIMENT_NAME" \
  agent="$AGENT_NAME" \
  agent.config.model_name_or_path="$MODEL_NAME_OR_PATH" \
  navsim_log_path="$NAVSIM_LOG_PATH_OVERRIDE" \
  original_sensor_path="$ORIGINAL_SENSOR_PATH_OVERRIDE" \
  synthetic_sensor_path="$SYNTHETIC_SENSOR_PATH_OVERRIDE" \
  synthetic_scenes_path="$SYNTHETIC_SCENES_PATH_OVERRIDE" \
  metric_cache_path="$CACHE_PATH_OVERRIDE" \
  worker=single_machine_thread_pool \
  worker.use_process_pool=True
