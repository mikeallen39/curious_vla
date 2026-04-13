#!/usr/bin/env bash

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/42_store/zxz/HUAWEI/VLA/curious_vla}"
export DATA_ROOT="${DATA_ROOT:-/data/zxz/HUAWEI/VLA/navsim_data}"

export CONDA_ROOT="/home/zxz/anaconda3"
export NAVSIM_ENV_PREFIX="/data/zxz/condaenv/curious_vla/navsim"
export LF_ENV_PREFIX="/data/zxz/condaenv/curious_vla/lf"
export LLAMAFACTORY_ROOT="/data/zxz/condaenv/curious_vla/src/LlamaFactory"
export CURIOUS_VLA_MODEL_DIR="/data/zxz/condaenv/curious_vla/models/Curious-VLA"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT/navsim_eval"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp_root"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export STATS_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"

export TRAIN_TEST_SPLIT="warmup_two_stage"
export CACHE_PATH="$NAVSIM_EXP_ROOT/metric_cache_warmup_two_stage"

export WARMUP_ROOT="$DATA_ROOT/warmup_two_stage"
export WARMUP_SYNTHETIC_SCENES_PATH="$WARMUP_ROOT/synthetic_scene_pickles"
export TEST_LOG_PATH="$DATA_ROOT/navsim_logs/test"
export ORIGINAL_SENSOR_PATH="$DATA_ROOT/sensor_blobs/test"
export SYNTHETIC_SENSOR_PATH="$WARMUP_ROOT/sensor_blobs"
