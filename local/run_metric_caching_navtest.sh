#!/usr/bin/env bash
set -euo pipefail

source /mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/local_env.sh
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$NAVSIM_ENV_PREFIX"

mkdir -p "$NAVSIM_EXP_ROOT"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split="$TRAIN_TEST_SPLIT" \
  navsim_log_path="$TEST_LOG_PATH" \
  original_sensor_path="$ORIGINAL_SENSOR_PATH" \
  synthetic_sensor_path="$SYNTHETIC_SENSOR_PATH" \
  synthetic_scenes_path="$WARMUP_SYNTHETIC_SCENES_PATH" \
  metric_cache_path="$CACHE_PATH"
