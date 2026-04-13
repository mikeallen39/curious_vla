#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/local_env.sh"

DOWNLOAD_ROOT="$DATA_ROOT/downloads"

if [ -d "$DOWNLOAD_ROOT/maps" ]; then
  rm -rf "$DATA_ROOT/maps"
  mv "$DOWNLOAD_ROOT/maps" "$DATA_ROOT/maps"
fi

if [ -d "$DOWNLOAD_ROOT/warmup_two_stage" ]; then
  rm -rf "$DATA_ROOT/warmup_two_stage"
  mv "$DOWNLOAD_ROOT/warmup_two_stage" "$DATA_ROOT/warmup_two_stage"
fi

if [ -d "$DOWNLOAD_ROOT/test_navsim_logs" ]; then
  mkdir -p "$DATA_ROOT/navsim_logs"
  if [ -d "$DOWNLOAD_ROOT/test_navsim_logs/test" ]; then
    rm -rf "$DATA_ROOT/navsim_logs/test"
    mv "$DOWNLOAD_ROOT/test_navsim_logs/test" "$DATA_ROOT/navsim_logs/test"
    rmdir "$DOWNLOAD_ROOT/test_navsim_logs" 2>/dev/null || true
  else
    rm -rf "$DATA_ROOT/navsim_logs/test"
    mv "$DOWNLOAD_ROOT/test_navsim_logs" "$DATA_ROOT/navsim_logs/test"
  fi
fi

mkdir -p "$NAVSIM_EXP_ROOT"

echo "warmup data layout prepared under $DATA_ROOT"
