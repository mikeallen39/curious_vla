#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMAFACTORY_ROOT="$PROJECT_ROOT/LLaMA-Factory"
LLAMAFACTORY_REPO_URL="https://github.com/hiyouga/LLaMA-Factory.git"

if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  bash scripts/setup_env.sh

This script:
  - creates `curious` (Python 3.10)
  - installs EasyR1 into `curious`
  - clones `LLaMA-Factory/` if missing, then installs it into `curious`
  - creates `navsim` from `navsim_eval/environment.yml`
  - installs `navsim_eval` into `navsim`

Data setup is still manual. See:
  - `navsim_eval/docs/install.md`
  - `docs/deploy.md`
  - `docs/train_sft.md`
EOF
    exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Setting up curious..."
if ! conda env list | grep -qw "^curious"; then
    conda create -n curious python=3.10 -y
fi

conda activate curious
pip install -e "$PROJECT_ROOT/EasyR1"

if [[ ! -d "$LLAMAFACTORY_ROOT/.git" ]]; then
    if [[ -e "$LLAMAFACTORY_ROOT" ]]; then
        echo "Cannot clone LLaMA-Factory into: $LLAMAFACTORY_ROOT" >&2
        exit 1
    fi
    git clone --depth 1 "$LLAMAFACTORY_REPO_URL" "$LLAMAFACTORY_ROOT"
fi

pip install -e "$LLAMAFACTORY_ROOT"
conda deactivate

echo "Setting up navsim..."
if ! conda env list | grep -qw "^navsim"; then
    conda env create --name navsim -f "$PROJECT_ROOT/navsim_eval/environment.yml"
fi

conda activate navsim
pip install -e "$PROJECT_ROOT/navsim_eval"
python -c "import torch" >/dev/null 2>&1 || pip install torch
conda deactivate

echo "Setup complete."
echo "Optional: conda activate curious && pip install flash-attn --no-build-isolation"
