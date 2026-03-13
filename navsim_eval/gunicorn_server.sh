#!/bin/bash
pkill -9 gunicorn

# ===== User Configuration (modify before running) =====
PROJECT_ROOT="/path/to/curious_vla"
DATA_ROOT="/path/to/navsim_data"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT/navsim_eval"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp_root"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_train

HOST="0.0.0.0"
PORT=8901

NUM_WORKERS=$(nproc) # set by num of CPU cores, adjust as needed

echo "[INFO] Starting Gunicorn server on $HOST:$PORT with $NUM_WORKERS workers..."

gunicorn navsim.planning.script.run_gunicorn_server:app \
    -w $NUM_WORKERS \
    -k uvicorn.workers.UvicornWorker \
    -b $HOST:$PORT \
    --timeout 150 \
    --log-level info \

echo "[INFO] Gunicorn server started."
