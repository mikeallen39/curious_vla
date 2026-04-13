#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/local_env.sh"

: "${WATCH_INTERVAL:=60}"
: "${GPU_SMOKE_PORT:=8192}"
: "${GPU_SMOKE_MAX_TOKENS:=512}"
: "${GPU_SMOKE_EXPERIMENT_PREFIX:=warmup_gpu_watch}"
: "${GPU_SMOKE_ALLOWED_GPUS:=0,1,2,3}"

WATCH_LOG_DIR="$DATA_ROOT/logs"
mkdir -p "$WATCH_LOG_DIR" "$NAVSIM_EXP_ROOT"

timestamp="$(date +%Y%m%d_%H%M%S)"
watch_log="$WATCH_LOG_DIR/${GPU_SMOKE_EXPERIMENT_PREFIX}_${timestamp}.log"
server_log="$NAVSIM_EXP_ROOT/${GPU_SMOKE_EXPERIMENT_PREFIX}_server_${timestamp}.log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$watch_log"
}

current_original_sensor_root() {
  if [ -d "$DATA_ROOT/sensor_blobs/test" ]; then
    printf '%s\n' "$DATA_ROOT/sensor_blobs/test"
    return 0
  fi
  if [ -d "$DATA_ROOT/downloads/test_sensor_blobs/test" ]; then
    printf '%s\n' "$DATA_ROOT/downloads/test_sensor_blobs/test"
    return 0
  fi
  if [ -d "$DATA_ROOT/downloads/openscene-v1.1/sensor_blobs/test" ]; then
    printf '%s\n' "$DATA_ROOT/downloads/openscene-v1.1/sensor_blobs/test"
    return 0
  fi
  return 1
}

pick_available_warmup_original() {
  local sensor_root="$1"
  "$NAVSIM_ENV_PREFIX/bin/python" - <<'PY' "$TEST_LOG_PATH" "$sensor_root" "$PROJECT_ROOT"
import pickle
import sys
from pathlib import Path

test_log_root = Path(sys.argv[1])
sensor_root = Path(sys.argv[2])
project_root = Path(sys.argv[3])
yaml_path = project_root / "navsim_eval/navsim/planning/script/config/common/train_test_split/scene_filter/warmup_two_stage.yaml"

log_names = []
current = None
with open(yaml_path, "r", encoding="utf-8") as f:
    for raw in f:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if stripped == "log_names:":
            current = "log_names"
            continue
        if current == "log_names":
            if stripped.startswith("- "):
                log_names.append(stripped[2:].strip().strip("'").strip('"'))
                continue
            current = None

for log_name in log_names:
    log_pickle = test_log_root / f"{log_name}.pkl"
    if not log_pickle.exists():
        continue
    with open(log_pickle, "rb") as f:
        frames = pickle.load(f)
    for idx in range(3, len(frames), 1):
        token = frames[idx]["token"]
        rel = frames[idx]["cams"]["CAM_F0"]["data_path"]
        image_path = sensor_root / rel
        if image_path.exists():
            print(f"{log_name}|{token}|{rel}")
            raise SystemExit(0)

raise SystemExit(1)
PY
}

pick_gpu() {
  local allowed_gpus="$GPU_SMOKE_ALLOWED_GPUS"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' -v allowed="$allowed_gpus" '
      BEGIN {
        n = split(allowed, ids, ",")
        for (i = 1; i <= n; ++i) {
          gsub(/^[ \t]+|[ \t]+$/, "", ids[i])
          ok[ids[i]] = 1
        }
      }
      {
        idx = $1
        free = $2
        gsub(/^[ \t]+|[ \t]+$/, "", idx)
        gsub(/^[ \t]+|[ \t]+$/, "", free)
        if (idx in ok) {
          print idx "," free
        }
      }
    ' \
    | sort -t',' -k2 -nr \
    | head -n 1 \
    | cut -d',' -f1
}

wait_for_server() {
  local tries=0
  while ! curl -sf "http://127.0.0.1:${GPU_SMOKE_PORT}/v1/models" >/dev/null; do
    tries=$((tries + 1))
    if [ "$tries" -gt 120 ]; then
      log "server did not become ready on port ${GPU_SMOKE_PORT}"
      return 1
    fi
    sleep 5
  done
}

log "watcher started"
log "watch log: $watch_log"

selection=""
sensor_root=""
while true; do
  if sensor_root="$(current_original_sensor_root)"; then
    if selection="$(pick_available_warmup_original "$sensor_root" 2>/dev/null)"; then
      break
    fi
    log "test camera root exists at $sensor_root but no warmup original token is readable yet"
  else
    log "test camera root is not available yet"
  fi
  sleep "$WATCH_INTERVAL"
done

IFS='|' read -r selected_log selected_token selected_relpath <<<"$selection"
log "detected warmup original token"
log "log=$selected_log token=$selected_token image=$selected_relpath sensor_root=$sensor_root"

gpu_id="$(pick_gpu)"
if [ -z "$gpu_id" ]; then
  log "no GPU available within allowed set: ${GPU_SMOKE_ALLOWED_GPUS}"
  exit 1
fi
log "selected gpu=$gpu_id"

server_pid=""
if curl -sf "http://127.0.0.1:${GPU_SMOKE_PORT}/v1/models" >/dev/null; then
  log "reusing existing server on port ${GPU_SMOKE_PORT}"
else
  log "starting llamafactory server on gpu=$gpu_id port=${GPU_SMOKE_PORT}"
  nohup bash -lc "
    source '$CONDA_ROOT/etc/profile.d/conda.sh'
    conda activate '$LF_ENV_PREFIX'
    cd '$PROJECT_ROOT/navsim_eval'
    export CUDA_VISIBLE_DEVICES='$gpu_id' API_HOST=127.0.0.1 API_PORT='${GPU_SMOKE_PORT}' API_VERBOSE=0
    exec llamafactory-cli api \
      --model_name_or_path '$CURIOUS_VLA_MODEL_DIR' \
      --template qwen2_vl \
      --infer_backend huggingface \
      --image_max_pixels 262144 \
      --trust_remote_code true
  " >"$server_log" 2>&1 &
  server_pid=$!
  log "server pid=$server_pid server_log=$server_log"
  wait_for_server
fi

run_tag="$(date +%Y%m%d_%H%M%S)"
cache_dir="$NAVSIM_EXP_ROOT/metric_cache_${GPU_SMOKE_EXPERIMENT_PREFIX}_${run_tag}"
experiment_name="${GPU_SMOKE_EXPERIMENT_PREFIX}_${run_tag}"

log "building metric cache at $cache_dir"
"$NAVSIM_ENV_PREFIX/bin/python" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split="$TRAIN_TEST_SPLIT" \
  "train_test_split.scene_filter.log_names=['${selected_log}']" \
  train_test_split.scene_filter.max_scenes=1 \
  train_test_split.scene_filter.include_synthetic_scenes=false \
  navsim_log_path="$TEST_LOG_PATH" \
  original_sensor_path="$sensor_root" \
  synthetic_sensor_path="$SYNTHETIC_SENSOR_PATH" \
  synthetic_scenes_path="$WARMUP_SYNTHETIC_SCENES_PATH" \
  metric_cache_path="$cache_dir" \
  worker=sequential | tee -a "$watch_log"

log "running one-scene gpu eval experiment_name=$experiment_name"
"$NAVSIM_ENV_PREFIX/bin/python" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py" \
  train_test_split="$TRAIN_TEST_SPLIT" \
  "train_test_split.scene_filter.log_names=['${selected_log}']" \
  train_test_split.scene_filter.max_scenes=1 \
  train_test_split.scene_filter.include_synthetic_scenes=false \
  experiment_name="$experiment_name" \
  agent=navsim_qwen_norm_cot_baseline_agent \
  "agent.config.model_name_or_path=$CURIOUS_VLA_MODEL_DIR" \
  "+agent.config.api_base_url=http://127.0.0.1:${GPU_SMOKE_PORT}/v1" \
  "+agent.config.max_tokens=${GPU_SMOKE_MAX_TOKENS}" \
  +agent.config.temperature=0.0 \
  navsim_log_path="$TEST_LOG_PATH" \
  original_sensor_path="$sensor_root" \
  synthetic_sensor_path="$SYNTHETIC_SENSOR_PATH" \
  synthetic_scenes_path="$WARMUP_SYNTHETIC_SCENES_PATH" \
  metric_cache_path="$cache_dir" \
  worker=sequential | tee -a "$watch_log"

if [ -n "$server_pid" ]; then
  log "stopping temporary server pid=$server_pid"
  kill "$server_pid" || true
fi

log "watcher finished"
