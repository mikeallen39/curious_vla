#!/usr/bin/env bash
set -euo pipefail

# ========== 通用单次评估 wrapper ==========
# 启动 LF 服务 → 等待就绪 → 评估 → 清理
#
# 必需环境变量（由外层 scripts/run_eval.sh 设置）:
#   MODEL_NAME_OR_PATH, EXPERIMENT_NAME
#   NAVSIM_DEVKIT_ROOT, OPENSCENE_DATA_ROOT, NAVSIM_EXP_ROOT,
#   NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION

NAVSIM_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

AGENT_NAME="${AGENT_NAME:-navsim_qwen_norm_cot_baseline_agent}"
TEMPLATE="${TEMPLATE:-qwen2_vl}"
NUM_INSTANCES="${NUM_INSTANCES:-8}"
START_PORT="${START_PORT:-8192}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
WARMUP_SLEEP="${WARMUP_SLEEP:-20}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
: "${NAVSIM_EXP_ROOT:?Error: NAVSIM_EXP_ROOT is not set}"
CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_${TRAIN_TEST_SPLIT}}"
TOTAL_GPUS="${TOTAL_GPUS:-$NUM_INSTANCES}"

if [ -z "$MODEL_NAME_OR_PATH" ]; then
    echo "Error: MODEL_NAME_OR_PATH is not set"; exit 1
fi
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Error: EXPERIMENT_NAME is not set"; exit 1
fi
if [ ! -d "$CACHE_PATH" ]; then
    echo "Error: metric cache not found: $CACHE_PATH"
    echo "Set TRAIN_TEST_SPLIT=\"$TRAIN_TEST_SPLIT\" in navsim_eval/scripts/evaluation/run_metric_caching.sh, then run that script first."
    exit 1
fi

# ========== 构建端口列表 ==========
ports=()
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    ports+=($((START_PORT + i)))
done

SESSION_NAME="vllm_eval_$$"

# ========== 清理函数 ==========
cleanup_vllm() {
    echo "[cleanup] Stopping serving ..."
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
}
trap cleanup_vllm EXIT

# ========== 启动 LF 服务 ==========
echo "=========================================="
echo " Starting LF serving: $MODEL_NAME_OR_PATH"
echo " EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo " AGENT_NAME=$AGENT_NAME"
echo "=========================================="

tmux new-session -d -s "$SESSION_NAME" \
    "bash -c 'model_name_or_path=$MODEL_NAME_OR_PATH template=$TEMPLATE TOTAL_GPUS=$TOTAL_GPUS bash $NAVSIM_ROOT/lf_serve_cot.sh $NUM_INSTANCES 2>&1 | tee /tmp/${SESSION_NAME}.log'"

# ========== 等待服务就绪 ==========
echo "[wait] Waiting for serving instances ..."
ready=false
for i in $(seq 1 "$WAIT_TIMEOUT"); do
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "[error] Serving session exited early. Check /tmp/${SESSION_NAME}.log"
        exit 1
    fi
    all_up=true
    for p in "${ports[@]}"; do
        if ! lsof -i :"$p" &>/dev/null; then
            all_up=false
            break
        fi
    done
    if $all_up; then
        echo "[wait] All ports ready."
        ready=true
        break
    fi
    sleep 5
done

if [ "$ready" = false ]; then
    echo "[error] Server startup timed out."
    exit 1
fi

sleep "$WARMUP_SLEEP"

# ========== 执行评估 ==========
echo "[eval] Running evaluation ..."

python "$NAVSIM_ROOT/navsim/planning/script/run_pdm_score_one_stage.py" \
    train_test_split="$TRAIN_TEST_SPLIT" \
    experiment_name="$EXPERIMENT_NAME" \
    agent="$AGENT_NAME" \
    agent.config.model_name_or_path="$MODEL_NAME_OR_PATH" \
    metric_cache_path="$CACHE_PATH" \
    worker=single_machine_thread_pool \
    worker.use_process_pool=True

echo "[done] Evaluation complete: $EXPERIMENT_NAME"
