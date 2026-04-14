#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

detect_device_type() {
  if [[ "${LOCAL_DEVICE_TYPE:-}" == "gpu" || "${LOCAL_DEVICE_TYPE:-}" == "npu" ]]; then
    printf '%s\n' "$LOCAL_DEVICE_TYPE"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    printf '%s\n' "gpu"
    return 0
  fi
  if command -v npu-smi >/dev/null 2>&1 && npu-smi info >/dev/null 2>&1; then
    printf '%s\n' "npu"
    return 0
  fi
  echo "Unable to detect device type. Please pass --device-type gpu|npu." >&2
  exit 1
}

backend="hf"
device_type="auto"
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
    --device-type)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --device-type" >&2
        exit 1
      fi
      device_type="$2"
      shift 2
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

case "$device_type" in
  auto)
    device_type="$(detect_device_type)"
    ;;
  gpu|npu)
    ;;
  *)
    echo "Unsupported device type: $device_type" >&2
    echo "Expected one of: auto, gpu, npu" >&2
    exit 1
    ;;
esac

case "$device_type" in
  gpu)
    source "$LOCAL_ROOT/gpu/local_env_gpu.sh"
    ;;
  npu)
    source "$LOCAL_ROOT/npu/local_env_npu.sh"
    if [[ "$backend" == "vllm" ]]; then
      source "$LOCAL_ROOT/npu/local_env_vllm_ascend.sh"
    fi
    ;;
esac

source "$CONDA_ROOT/etc/profile.d/conda.sh"

case "$backend" in
  hf|transformers|local)
    env_prefix="${LOCAL_BENCHMARK_ENV_PREFIX:-${LATENCY_ENV_PREFIX:-${LF_ENV_PREFIX:-${NAVSIM_ENV_PREFIX:-}}}}"
    ;;
  lf)
    env_prefix="${LF_BENCHMARK_ENV_PREFIX:-${LF_ENV_PREFIX:-${NAVSIM_ENV_PREFIX:-${LATENCY_ENV_PREFIX:-}}}}"
    ;;
  vllm)
    env_prefix="${VLLM_BENCHMARK_ENV_PREFIX:-${VLLM_ASCEND_ENV_PREFIX:-${LF_ENV_PREFIX:-${NAVSIM_ENV_PREFIX:-${LATENCY_ENV_PREFIX:-}}}}}"
    ;;
  *)
    echo "Unsupported backend: $backend" >&2
    echo "Expected one of: hf, transformers, local, lf, vllm" >&2
    exit 1
    ;;
esac

if [[ -z "${env_prefix:-}" ]]; then
  echo "No conda environment configured for backend=$backend device_type=$device_type" >&2
  exit 1
fi

conda activate "$env_prefix"

latency_log_dir="${LATENCY_LOG_DIR:-${NAVSIM_EXP_ROOT:-$PROJECT_ROOT/exp_root}/latency}"
mkdir -p "$latency_log_dir"

scene_limit="${SCENE_LIMIT:-4}"
warmup_runs="${WARMUP_RUNS:-1}"
benchmark_runs="${BENCHMARK_RUNS:-4}"
width="${WIDTH:-1280}"
height="${HEIGHT:-704}"
selection_mode="${SELECTION_MODE:-diverse-by-command}"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_json="${OUTPUT_JSON:-$latency_log_dir/planning_latency_${backend}_${device_type}_${timestamp}.json}"

cmd=(
  python "$SCRIPT_DIR/run_planning_latency_benchmark.py"
  --backend "$backend"
  --device-type "$device_type"
  --model "$CURIOUS_VLA_MODEL_DIR"
  --warmup-root "$WARMUP_ROOT"
  --scene-limit "$scene_limit"
  --warmup-runs "$warmup_runs"
  --benchmark-runs "$benchmark_runs"
  --width "$width"
  --height "$height"
  --selection-mode "$selection_mode"
  --output-json "$output_json"
)

case "$backend" in
  lf)
    base_url="${BASE_URL:-${LF_BASE_URL:-http://127.0.0.1:8192/v1}}"
    cmd+=(--base-url "$base_url" --run-text-planning-control)
    ;;
  vllm)
    base_url="${BASE_URL:-${VLLM_ASCEND_BASE_URL:-${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}}}"
    cmd+=(--base-url "$base_url" --run-text-planning-control)
    ;;
esac

exec "${cmd[@]}" "${args[@]}"
