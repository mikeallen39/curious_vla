#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
NAVSIM_DEVKIT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Edit before running.
DATA_ROOT="$PROJECT_ROOT/datasets/navsim"
TRAIN_TEST_SPLIT="navtest"
CACHE_PATH="$PROJECT_ROOT/exp_root/metric_cache_${TRAIN_TEST_SPLIT}"
EXPERIMENT_NAME="demo_test"
AGENT_NAME="navsim_qwen_norm_cot_baseline_agent"
MODEL_NAME_OR_PATH="none"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "DATA_ROOT not found: $DATA_ROOT" >&2
    exit 1
fi
if [[ ! -d "$CACHE_PATH" ]]; then
    echo "Metric cache not found: $CACHE_PATH" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate navsim

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="$NAVSIM_DEVKIT_ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp_root"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export STATS_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py" \
    train_test_split="$TRAIN_TEST_SPLIT" \
    experiment_name="$EXPERIMENT_NAME" \
    agent="$AGENT_NAME" \
    agent.config.model_name_or_path="$MODEL_NAME_OR_PATH" \
    metric_cache_path="$CACHE_PATH" \
    worker=single_machine_thread_pool \
    worker.use_process_pool=True
