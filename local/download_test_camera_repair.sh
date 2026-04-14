#!/usr/bin/env bash
set -euo pipefail

ROOT=/cache/ma-user/curious_vla_assets/data/downloads/test_sensor_blobs/openscene-v1.1/sensor_blobs
TARGET="$ROOT/test"
META_ROOT=/cache/ma-user/curious_vla_assets/data/downloads/test_navsim_logs/test
TMP=/cache/ma-user/curious_vla_assets/data/downloads/.repair_test_camera_20260413
DONE_DIR="$TMP/done"
PARTS_DIR="$TMP/parts"
PARALLEL_RANGES=8
PARALLEL_WORKERS=4

remaining() {
python - <<'PY'
from pathlib import Path

meta = {p.stem for p in Path("/cache/ma-user/curious_vla_assets/data/downloads/test_navsim_logs/test").glob("*.pkl")}
sensor = {
    p.name
    for p in Path("/cache/ma-user/curious_vla_assets/data/downloads/test_sensor_blobs/openscene-v1.1/sensor_blobs/test").glob("*")
    if p.is_dir()
}
print(len(meta - sensor))
PY
}

sample_missing() {
python - <<'PY'
from pathlib import Path

meta = {p.stem for p in Path("/cache/ma-user/curious_vla_assets/data/downloads/test_navsim_logs/test").glob("*.pkl")}
sensor = {
    p.name
    for p in Path("/cache/ma-user/curious_vla_assets/data/downloads/test_sensor_blobs/openscene-v1.1/sensor_blobs/test").glob("*")
    if p.is_dir()
}
missing = sorted(meta - sensor)
print(missing[:10])
PY
}

mkdir -p "$TMP/extract" "$DONE_DIR"

echo "start $(date '+%F %T')"
echo "meta_root=$META_ROOT"
echo "target=$TARGET"
echo "tmp=$TMP"
echo "remaining_before=$(remaining)"
echo "sample_missing_before=$(sample_missing)"

for split in $(seq 0 31); do
  if [ -f "$DONE_DIR/split_${split}.done" ]; then
    echo "skip_done_split=$split"
    continue
  fi

  rem=$(remaining)
  echo "split=$split remaining_before=$rem time=$(date '+%F %T')"
  if [ "$rem" -eq 0 ]; then
    echo "all_missing_logs_filled_before_split=$split"
    break
  fi

  file="$TMP/openscene_sensor_test_camera_${split}.tgz"
  url="https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_${split}.tgz"

  python /home/ma-user/curious_vla/local/download_http_ranges.py \
    "$url" \
    "$file" \
    --parts "$PARALLEL_RANGES" \
    --concurrency "$PARALLEL_WORKERS" \
    --part-dir "$PARTS_DIR/split_${split}"
  rm -rf "$TMP/extract/openscene-v1.1"
  tar -xzf "$file" -C "$TMP/extract"
  mkdir -p "$ROOT/test"
  for src in "$TMP/extract/openscene-v1.1/sensor_blobs/test"/*; do
    name=$(basename "$src")
    dst="$ROOT/test/$name"
    if [ -e "$dst" ]; then
      echo "skip_existing_log=$name"
      continue
    fi
    mv "$src" "$dst"
    echo "added_missing_log=$name"
  done
  rm -rf "$TMP/extract/openscene-v1.1"
  rm -f "$file"
  touch "$DONE_DIR/split_${split}.done"

  echo "split=$split remaining_after=$(remaining) time=$(date '+%F %T')"
done

echo "final_remaining=$(remaining)"
echo "sample_missing_after=$(sample_missing)"
echo "end $(date '+%F %T')"
