#!/usr/bin/env bash

LOCAL_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

detect_device_type() {
  if [ -n "${LOCAL_DEVICE_TYPE:-}" ]; then
    printf '%s\n' "$LOCAL_DEVICE_TYPE"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    printf 'gpu\n'
    return 0
  fi

  if command -v npu-smi >/dev/null 2>&1 || [ -e /dev/davinci_manager ] || [ -d /usr/local/Ascend ]; then
    printf 'npu\n'
    return 0
  fi

  printf 'gpu\n'
}

case "$(detect_device_type)" in
  gpu)
    # shellcheck disable=SC1091
    source "$LOCAL_ENV_DIR/gpu/local_env_gpu.sh"
    ;;
  npu)
    # shellcheck disable=SC1091
    source "$LOCAL_ENV_DIR/npu/local_env_npu.sh"
    ;;
  *)
    echo "Unsupported LOCAL_DEVICE_TYPE=${LOCAL_DEVICE_TYPE:-unknown}" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
