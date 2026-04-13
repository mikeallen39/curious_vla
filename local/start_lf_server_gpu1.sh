#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/local_env.sh"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$LF_ENV_PREFIX"

cd "$PROJECT_ROOT/navsim_eval"

: "${model_name_or_path:=$CURIOUS_VLA_MODEL_DIR}"
: "${template:=qwen2_vl}"
: "${infer_backend:=huggingface}"
: "${api_port:=8192}"

CUDA_VISIBLE_DEVICES=1 \
API_VERBOSE=0 \
API_PORT="$api_port" \
llamafactory-cli api \
  --model_name_or_path "$model_name_or_path" \
  --template "$template" \
  --infer_backend "$infer_backend" \
  --image_max_pixels 262144 \
  --trust_remote_code true
