#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/local_env.sh"

: "${API_BASE:=http://127.0.0.1:8192}"
: "${CACHE_CHECK_INTERVAL:=15}"
: "${API_CHECK_INTERVAL:=10}"
: "${EXPERIMENT_PREFIX:=warmup_smoke}"
: "${MODEL_NAME_OR_PATH:=$CURIOUS_VLA_MODEL_DIR}"

cache_pattern='[r]un_metric_caching.py'
cache_split_pattern="train_test_split=${TRAIN_TEST_SPLIT}"

while pgrep -af "$cache_pattern" | grep -F "$cache_split_pattern" >/dev/null; do
  sleep "$CACHE_CHECK_INTERVAL"
done

until curl -sf "${API_BASE}/v1/models" >/dev/null; do
  sleep "$API_CHECK_INTERVAL"
done

EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_$(date +%Y%m%d_%H%M%S)"
export EXPERIMENT_NAME
export MODEL_NAME_OR_PATH

exec /mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/run_warmup_eval.sh
