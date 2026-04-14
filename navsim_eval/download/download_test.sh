#!/usr/bin/env bash
set -euo pipefail

: "${DOWNLOAD_METADATA:=1}"
: "${DOWNLOAD_CAMERA:=1}"
: "${DOWNLOAD_LIDAR:=1}"
: "${SPLIT_START:=0}"
: "${SPLIT_END:=31}"

download_and_extract() {
    local url="$1"
    local archive_name="$2"

    wget -c "$url"
    echo "Extracting file ${archive_name}"
    tar -xzf "$archive_name"
    rm -f "$archive_name"
}

if [ "$DOWNLOAD_METADATA" = "1" ] && [ ! -d test_navsim_logs ] && [ ! -d test_navsim_logs/test ]; then
    download_and_extract \
        "https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz" \
        "openscene_metadata_test.tgz"
fi

if [ "$DOWNLOAD_CAMERA" = "1" ]; then
    for split in $(seq "$SPLIT_START" "$SPLIT_END"); do
        download_and_extract \
            "https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_${split}.tgz" \
            "openscene_sensor_test_camera_${split}.tgz"
    done
fi

if [ "$DOWNLOAD_LIDAR" = "1" ]; then
    for split in $(seq "$SPLIT_START" "$SPLIT_END"); do
        download_and_extract \
            "https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_lidar/openscene_sensor_test_lidar_${split}.tgz" \
            "openscene_sensor_test_lidar_${split}.tgz"
    done
fi

if [ -d openscene-v1.1/meta_datas ] && [ ! -d test_navsim_logs ]; then
    mv openscene-v1.1/meta_datas test_navsim_logs
fi

if [ -d openscene-v1.1/sensor_blobs ] && [ ! -d test_sensor_blobs ]; then
    mv openscene-v1.1/sensor_blobs test_sensor_blobs
fi

if [ -d openscene-v1.1 ] && [ -z "$(find openscene-v1.1 -mindepth 1 -print -quit)" ]; then
    rmdir openscene-v1.1
fi
