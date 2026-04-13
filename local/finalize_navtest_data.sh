#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/local_env.sh"

DOWNLOAD_ROOT="$DATA_ROOT/downloads"

mkdir -p "$DATA_ROOT/navsim_logs" "$DATA_ROOT/sensor_blobs"

if [ -d "$DOWNLOAD_ROOT/test_navsim_logs/test" ]; then
  rm -rf "$DATA_ROOT/navsim_logs/test"
  mv "$DOWNLOAD_ROOT/test_navsim_logs/test" "$DATA_ROOT/navsim_logs/test"
  rmdir "$DOWNLOAD_ROOT/test_navsim_logs" 2>/dev/null || true
elif [ -d "$DOWNLOAD_ROOT/test_navsim_logs" ]; then
  rm -rf "$DATA_ROOT/navsim_logs/test"
  mv "$DOWNLOAD_ROOT/test_navsim_logs" "$DATA_ROOT/navsim_logs/test"
fi

if [ -d "$DOWNLOAD_ROOT/test_sensor_blobs" ]; then
  rm -rf "$DATA_ROOT/sensor_blobs/test"
  mv "$DOWNLOAD_ROOT/test_sensor_blobs" "$DATA_ROOT/sensor_blobs/test"
fi

if [ -d "$DOWNLOAD_ROOT/maps" ]; then
  rm -rf "$DATA_ROOT/maps"
  mv "$DOWNLOAD_ROOT/maps" "$DATA_ROOT/maps"
fi

mkdir -p "$NAVSIM_EXP_ROOT"

echo "navtest data layout prepared under $DATA_ROOT"
