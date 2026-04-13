#!/usr/bin/env bash

# Copy this file to local/local_env_override.sh on a new machine and adjust paths.

export PROJECT_ROOT="/path/to/curious_vla"
export DATA_ROOT="/path/to/navsim_data"

export CONDA_ROOT="/path/to/conda"
export NAVSIM_ENV_PREFIX="/path/to/conda/env/navsim"
export LF_ENV_PREFIX="/path/to/conda/env/lf"
export LLAMAFACTORY_ROOT="/path/to/LlamaFactory"
export CURIOUS_VLA_MODEL_DIR="/path/to/Curious-VLA"

export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT/navsim_eval"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp_root"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export STATS_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"

export TRAIN_TEST_SPLIT="warmup_two_stage"
export CACHE_PATH="$NAVSIM_EXP_ROOT/metric_cache_warmup_two_stage"

export WARMUP_ROOT="$DATA_ROOT/warmup_two_stage"
export WARMUP_SYNTHETIC_SCENES_PATH="$WARMUP_ROOT/synthetic_scene_pickles"
export TEST_LOG_PATH="$DATA_ROOT/navsim_logs/test"
export ORIGINAL_SENSOR_PATH="$DATA_ROOT/sensor_blobs/test"
export SYNTHETIC_SENSOR_PATH="$WARMUP_ROOT/sensor_blobs"
