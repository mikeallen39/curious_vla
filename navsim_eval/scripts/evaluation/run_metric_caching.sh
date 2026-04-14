#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
NAVSIM_DEVKIT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Edit before running.
DATA_ROOT="$PROJECT_ROOT/datasets/navsim"
TRAIN_TEST_SPLIT="navtest"
CACHE_PATH="$PROJECT_ROOT/exp_root/metric_cache_${TRAIN_TEST_SPLIT}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "DATA_ROOT not found: $DATA_ROOT" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate navsim

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="$NAVSIM_DEVKIT_ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp_root"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split="$TRAIN_TEST_SPLIT" \
    metric_cache_path="$CACHE_PATH"
