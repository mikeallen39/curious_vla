#!/usr/bin/env bash

LOCAL_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "$LOCAL_ENV_DIR/.." && pwd)"
DEFAULT_OVERRIDE_PATH="$LOCAL_ENV_DIR/local_env_override.sh"

# Preferred workflow on a new machine:
# 1. export variables in the shell, or
# 2. create local/local_env_override.sh from local/local_env_override.example.sh
if [ -f "${LOCAL_ENV_OVERRIDE:-$DEFAULT_OVERRIDE_PATH}" ]; then
  # shellcheck disable=SC1090
  source "${LOCAL_ENV_OVERRIDE:-$DEFAULT_OVERRIDE_PATH}"
fi

if command -v conda >/dev/null 2>&1; then
  _detected_conda_root="$(conda info --base 2>/dev/null || true)"
else
  _detected_conda_root=""
fi

export PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
export DATA_ROOT="${DATA_ROOT:-/data/zxz/HUAWEI/VLA/navsim_data}"

export CONDA_ROOT="${CONDA_ROOT:-${_detected_conda_root:-/home/zxz/anaconda3}}"
export NAVSIM_ENV_PREFIX="${NAVSIM_ENV_PREFIX:-/data/zxz/condaenv/curious_vla/navsim}"
export LF_ENV_PREFIX="${LF_ENV_PREFIX:-/data/zxz/condaenv/curious_vla/lf}"
export LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/data/zxz/condaenv/curious_vla/src/LlamaFactory}"
export CURIOUS_VLA_MODEL_DIR="${CURIOUS_VLA_MODEL_DIR:-/data/zxz/condaenv/curious_vla/models/Curious-VLA}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$PROJECT_ROOT/navsim_eval}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$DATA_ROOT}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$PROJECT_ROOT/exp_root}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export STATS_PATH="${STATS_PATH:-$PROJECT_ROOT/stats/trajectory_stats_train.json}"

export TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-warmup_two_stage}"
export CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_warmup_two_stage}"

export WARMUP_ROOT="${WARMUP_ROOT:-$DATA_ROOT/warmup_two_stage}"
export WARMUP_SYNTHETIC_SCENES_PATH="${WARMUP_SYNTHETIC_SCENES_PATH:-$WARMUP_ROOT/synthetic_scene_pickles}"
export TEST_LOG_PATH="${TEST_LOG_PATH:-$DATA_ROOT/navsim_logs/test}"
export ORIGINAL_SENSOR_PATH="${ORIGINAL_SENSOR_PATH:-$DATA_ROOT/sensor_blobs/test}"
export SYNTHETIC_SENSOR_PATH="${SYNTHETIC_SENSOR_PATH:-$WARMUP_ROOT/sensor_blobs}"
