#!/usr/bin/env bash
set -euo pipefail

source /mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/local_env.sh

DOWNLOAD_ROOT="$DATA_ROOT/downloads"

if [ -d "$DOWNLOAD_ROOT/maps" ]; then
  ln -sfn "$DOWNLOAD_ROOT/maps" "$DATA_ROOT/maps"
fi

if [ -d "$DOWNLOAD_ROOT/warmup_two_stage" ]; then
  ln -sfn "$DOWNLOAD_ROOT/warmup_two_stage" "$DATA_ROOT/warmup_two_stage"
fi

if [ -d "$DOWNLOAD_ROOT/test_navsim_logs" ]; then
  mkdir -p "$DATA_ROOT/navsim_logs"
  if [ -d "$DOWNLOAD_ROOT/test_navsim_logs/test" ]; then
    ln -sfn "$DOWNLOAD_ROOT/test_navsim_logs/test" "$DATA_ROOT/navsim_logs/test"
  else
    ln -sfn "$DOWNLOAD_ROOT/test_navsim_logs" "$DATA_ROOT/navsim_logs/test"
  fi
fi

mkdir -p "$NAVSIM_EXP_ROOT"

echo "warmup data layout prepared under $DATA_ROOT"
