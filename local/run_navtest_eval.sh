#!/usr/bin/env bash
set -euo pipefail

source /mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/local_env.sh
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$NAVSIM_ENV_PREFIX"

: "${AGENT_NAME:=navsim_qwen_norm_cot_baseline_agent}"
: "${EXPERIMENT_NAME:=demo_test}"
: "${MODEL_NAME_OR_PATH:=$CURIOUS_VLA_MODEL_DIR}"

mkdir -p "$NAVSIM_EXP_ROOT"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py" \
  train_test_split="$TRAIN_TEST_SPLIT" \
  experiment_name="$EXPERIMENT_NAME" \
  agent="$AGENT_NAME" \
  agent.config.model_name_or_path="$MODEL_NAME_OR_PATH" \
  navsim_log_path="$TEST_LOG_PATH" \
  original_sensor_path="$WARMUP_SENSOR_PATH" \
  synthetic_sensor_path="$WARMUP_SENSOR_PATH" \
  synthetic_scenes_path="$WARMUP_SYNTHETIC_SCENES_PATH" \
  metric_cache_path="$CACHE_PATH" \
  worker=single_machine_thread_pool \
  worker.use_process_pool=True
