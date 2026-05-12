#!/usr/bin/env bash

export PROJECT_ROOT="/home/ma-user/curious_vla"
export ASSET_ROOT="/cache/ma-user/curious_vla_assets"

export CONDA_ROOT="/home/ma-user/anaconda3"
export LATENCY_ENV_PREFIX="$ASSET_ROOT/envs/curious-vla-npu-latency"
export CURIOUS_VLA_MODEL_DIR="$ASSET_ROOT/models/Curious-VLA"
export WARMUP_ROOT="$ASSET_ROOT/data/downloads/warmup_two_stage"
export LATENCY_LOG_DIR="$ASSET_ROOT/logs"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT/navsim_eval"
export OPENSCENE_DATA_ROOT="$WARMUP_ROOT"
export WARMUP_SENSOR_PATH="$WARMUP_ROOT/sensor_blobs"
export WARMUP_SYNTHETIC_SCENES_PATH="$WARMUP_ROOT/synthetic_scene_pickles"
export STATS_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
