#!/usr/bin/env bash
set -euo pipefail

backend="transformers"
args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --backend" >&2
        exit 1
      fi
      backend="$2"
      shift 2
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

case "$backend" in
  transformers)
    exec /home/ma-user/curious_vla/local/run_planning_latency_benchmark_npu.sh "${args[@]}"
    ;;
  vllm)
    exec /home/ma-user/curious_vla/local/run_vllm_planning_latency_benchmark.sh "${args[@]}"
    ;;
  *)
    echo "Unsupported backend: $backend" >&2
    echo "Expected one of: transformers, vllm" >&2
    exit 1
    ;;
esac
