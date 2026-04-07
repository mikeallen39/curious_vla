#!/usr/bin/env bash

export PROJECT_ROOT="/home/ma-user/curious_vla"
export ASSET_ROOT="/cache/ma-user/curious_vla_assets"

export CONDA_ROOT="/home/ma-user/anaconda3"
export VLLM_ASCEND_ENV_PREFIX="$ASSET_ROOT/envs/curious-vla-vllm-ascend"
export CURIOUS_VLA_MODEL_DIR="$ASSET_ROOT/models/Curious-VLA"
export WARMUP_ROOT="$ASSET_ROOT/data/downloads/warmup_two_stage"
export LATENCY_LOG_DIR="$ASSET_ROOT/logs"
export VLLM_ASCEND_BASE_URL="${VLLM_ASCEND_BASE_URL:-http://127.0.0.1:18002/v1}"
export STATS_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
