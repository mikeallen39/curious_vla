#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/local_env.sh"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$LF_ENV_PREFIX"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER="1"

MODEL_DIR="/data/zxz/condaenv/curious_vla/models/Curious-VLA"
mkdir -p "$MODEL_DIR"

while [ ! -f "$MODEL_DIR/model-00002-of-00002.safetensors" ]; do
  python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="MashiroLn/Curious-VLA", local_dir="/data/zxz/condaenv/curious_vla/models/Curious-VLA", max_workers=1)'
  sleep 5
done
