#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Edit before running.
DATA_ROOT="$PROJECT_ROOT/datasets/navsim"
MODEL_NAME_OR_PATH="/path/to/model"
EXPERIMENT_NAME="curious_vla_qwen2_5_vl_3b_sft_stage2"
NUM_GPUS=8
TRAIN_TEST_SPLIT="navtest"
AGENT_NAME="navsim_qwen_norm_cot_baseline_agent"
TEMPLATE="qwen2_vl"

if [[ "$MODEL_NAME_OR_PATH" == "/path/to/"* ]]; then
    echo "Set MODEL_NAME_OR_PATH first." >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "DATA_ROOT not found: $DATA_ROOT" >&2
    exit 1
fi

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT/navsim_eval"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp_root"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export MODEL_NAME_OR_PATH EXPERIMENT_NAME AGENT_NAME TEMPLATE TRAIN_TEST_SPLIT
export NUM_INSTANCES="$NUM_GPUS"
export CACHE_PATH="$NAVSIM_EXP_ROOT/metric_cache_${TRAIN_TEST_SPLIT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate navsim
bash "$PROJECT_ROOT/navsim_eval/scripts/evaluation/evaluate_single.sh"
