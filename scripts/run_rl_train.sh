#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Edit before running.
DATA_ROOT="$PROJECT_ROOT/datasets/navsim"
MODEL_PATH="/path/to/sft_model"
EXP_NAME="grpo_round1"
ADAS_FILTER_FILE="$PROJECT_ROOT/token_filters/curious_vla_qwen2_5_vl_3b_sft_stage2_adas1x_6k.txt"  # change this only if you want a custom filter
NUM_GPUS=8
REWARD_SERVER_PORT=8901
CACHE_SPLIT="navtrain"
# RESUME_FROM=""

NAVSIM_ROOT="$PROJECT_ROOT/navsim_eval"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
DATA_PATH="$EASYR1_ROOT/data/QA_navtrain_poutine_style_full"
REWARD_FN="$EASYR1_ROOT/verl/utils/reward_score/navsim/navsim_reward_text.py"
STATS_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
SESSION_NAME="curious_reward_$$"
CACHE_PATH="$PROJECT_ROOT/exp_root/metric_cache_${CACHE_SPLIT}"

if [[ "$MODEL_PATH" == "/path/to/"* ]]; then
    echo "Set MODEL_PATH first." >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "DATA_ROOT not found: $DATA_ROOT" >&2
    exit 1
fi
if [[ ! -f "$ADAS_FILTER_FILE" ]]; then
    echo "ADAS filter file not found: $ADAS_FILTER_FILE" >&2
    exit 1
fi
if [[ ! -d "$DATA_PATH" ]]; then
    echo "Training data not found: $DATA_PATH" >&2
    exit 1
fi
if [[ ! -d "$CACHE_PATH" ]]; then
    echo "Metric cache not found: $CACHE_PATH" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
trap 'tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true' EXIT

tmux new-session -d -s "$SESSION_NAME" \
    "bash -c '
        source \"$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate navsim
        export PROJECT_ROOT=\"$PROJECT_ROOT\"
        export DATA_ROOT=\"$DATA_ROOT\"
        export REWARD_SERVER_PORT=$REWARD_SERVER_PORT
        export CACHE_SPLIT=\"$CACHE_SPLIT\"
        export CACHE_PATH=\"$CACHE_PATH\"
        cd \"$NAVSIM_ROOT\"
        bash gunicorn_server.sh
    '"

for _ in $(seq 1 24); do
    tmux has-session -t "$SESSION_NAME" 2>/dev/null || {
        echo "Reward server exited early." >&2
        exit 1
    }
    lsof -i :"$REWARD_SERVER_PORT" >/dev/null 2>&1 && break
    sleep 5
done
lsof -i :"$REWARD_SERVER_PORT" >/dev/null 2>&1 || {
    echo "Reward server startup timed out." >&2
    exit 1
}
sleep 5

conda activate curious
cd "$EASYR1_ROOT"

export EXP_NAME NAVSIM_STAT_PATH="$STATS_PATH"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag

python3 -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files="${DATA_PATH}@train" \
    data.val_files="${DATA_PATH}@test" \
    data.format_prompt=null \
    data.max_response_length=3072 \
    worker.actor.model.model_path="$MODEL_PATH" \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.reward.reward_function="${REWARD_FN}:compute_score_fast" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.total_epochs=15 \
    ${ADAS_FILTER_FILE:+data.token_filter_file=$ADAS_FILTER_FILE} \
    ${RESUME_FROM:+trainer.load_checkpoint_path=$RESUME_FROM}
