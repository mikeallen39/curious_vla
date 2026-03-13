# ===== User Configuration (modify before running) =====
PROJECT_ROOT="/path/to/curious_vla"
DATA_ROOT="/path/to/navsim_data"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT/navsim_eval"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp_root"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"

TRAIN_TEST_SPLIT=navtest
# TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navtest
# CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
metric_cache_path=$CACHE_PATH
