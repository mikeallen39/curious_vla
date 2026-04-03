#!/usr/bin/env bash
set -euo pipefail

source /mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/local_env.sh

DOWNLOAD_ROOT="$DATA_ROOT/downloads"

mkdir -p "$DATA_ROOT/navsim_logs" "$DATA_ROOT/sensor_blobs"

if [ -d "$DOWNLOAD_ROOT/test_navsim_logs" ]; then
  ln -sfn "$DOWNLOAD_ROOT/test_navsim_logs" "$DATA_ROOT/navsim_logs/test"
fi

if [ -d "$DOWNLOAD_ROOT/test_sensor_blobs" ]; then
  ln -sfn "$DOWNLOAD_ROOT/test_sensor_blobs" "$DATA_ROOT/sensor_blobs/test"
fi

if [ -d "$DOWNLOAD_ROOT/maps" ]; then
  ln -sfn "$DOWNLOAD_ROOT/maps" "$DATA_ROOT/maps"
fi

mkdir -p "$NAVSIM_EXP_ROOT"

echo "navtest data layout prepared under $DATA_ROOT"
