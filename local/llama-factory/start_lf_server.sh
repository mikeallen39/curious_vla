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
: "${num_instances:=8}"

export model_name_or_path
export template

if [ "$infer_backend" = "vllm" ]; then
  bash ./lf_serve_cot.sh "$num_instances"
else
  API_VERBOSE=0 \
  llamafactory-cli api \
    --model_name_or_path "$model_name_or_path" \
    --template "$template" \
    --infer_backend "$infer_backend" \
    --image_max_pixels 262144 \
    --trust_remote_code true
fi
