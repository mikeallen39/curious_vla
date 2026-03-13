# ===== User Configuration (modify before running) =====
PROJECT_ROOT="/path/to/curious_vla"
DATA_ROOT="/path/to/navsim_data"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT/navsim_eval"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp_root"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navtest

SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

TRAIN_TEST_SPLIT=navtest

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
train_test_split=$TRAIN_TEST_SPLIT \
experiment_name=cv_navtest \
metric_cache_path=$CACHE_PATH \
worker=single_machine_thread_pool \
worker.use_process_pool=True \
# synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
# synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
