# ===== User Configuration (modify before running) =====
PROJECT_ROOT="/path/to/curious_vla"
DATA_ROOT="/path/to/navsim_data"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT/navsim_eval"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp_root"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navtest
export STATS_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"


TRAIN_TEST_SPLIT=navtest


: "${AGENT_NAME:=navsim_qwen_norm_cot_baseline_agent}"
: "${EXPERIMENT_NAME:=demo_test}"
: "${MODEL_NAME_OR_PATH:=YOUR_MODEL_PATH}" # if use lf api serve, can be none; if use vllm serve, should fill the actual path.

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
train_test_split=$TRAIN_TEST_SPLIT \
experiment_name=$EXPERIMENT_NAME \
agent=$AGENT_NAME \
agent.config.model_name_or_path=$MODEL_NAME_OR_PATH \
metric_cache_path=$CACHE_PATH \
worker=single_machine_thread_pool \
worker.use_process_pool=True
