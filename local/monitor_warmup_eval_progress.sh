#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/exp_root"

latest_log() {
  find "$ROOT" -path '*/log.txt' | sort | tail -n 1
}

LOG_PATH="${1:-}"
if [ -z "$LOG_PATH" ]; then
  LOG_PATH="$(latest_log)"
fi

if [ -z "$LOG_PATH" ] || [ ! -f "$LOG_PATH" ]; then
  echo "No evaluation log found."
  exit 1
fi

echo "log: $LOG_PATH"

awk '
  /Starting pdm scoring of [0-9]+ scenarios/ {
    if (match($0, /Starting pdm scoring of ([0-9]+) scenarios/, a)) {
      total = a[1]
    }
  }
  /Processing scenario [0-9]+ \/ [0-9]+ in thread_id=/ {
    if (match($0, /Processing scenario ([0-9]+) \/ ([0-9]+) in thread_id=([^,]+)/, a)) {
      idx = a[1] + 0
      cnt = a[2] + 0
      tid = a[3]
      seen[tid] = idx
      total_by_thread[tid] = cnt
    }
  }
  END {
    workers = 0
    started = 0
    completed = 0
    expected = 0

    for (tid in seen) {
      workers++
      started += seen[tid]
      completed += (seen[tid] - 1)
      expected += total_by_thread[tid]
    }

    if (total == 0 && expected > 0) {
      total = expected
    }

    printf("workers_seen: %d\n", workers)
    if (total > 0) {
      printf("total_scenarios: %d\n", total)
      printf("started_at_least: %d\n", started)
      printf("completed_at_least: %d\n", completed)
      printf("started_ratio_at_least: %.2f%%\n", 100.0 * started / total)
      printf("completed_ratio_at_least: %.2f%%\n", 100.0 * completed / total)
    } else {
      printf("total_scenarios: unknown\n")
      printf("started_at_least: %d\n", started)
      printf("completed_at_least: %d\n", completed)
    }

    for (tid in seen) {
      printf("thread %s: scenario %d / %d\n", tid, seen[tid], total_by_thread[tid])
    }
  }
' "$LOG_PATH"

echo
echo "processes:"
ps -ef | rg 'run_pdm_score_one_stage.py|llamafactory-cli api' || true

echo
echo "gpu:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,utilization.memory --format=csv,noheader
