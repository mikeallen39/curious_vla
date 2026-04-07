# Planning Latency Benchmark On Ascend NPU

This benchmark measures a path closer to end-to-end planning latency than the existing model-only latency script:

- load a real warmup scene pickle from `openscene_meta_datas`
- build `AgentInput` from the real history frames
- run `NavsimCoTQwenAgent.compute_trajectory()`
- use a local in-process `transformers + torch_npu` backend instead of an OpenAI-compatible server

## Environment

The benchmark uses the NPU environment and assets configured by:

- `local/local_env_npu.sh`

Important paths:

- model: `/cache/ma-user/curious_vla_assets/models/Curious-VLA`
- warmup data: `/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage`
- env: `/cache/ma-user/curious_vla_assets/envs/curious-vla-npu-latency`

## Commands

Fixed resize, closer to the previous 1280x704 model-only benchmark:

```bash
./local/run_planning_latency_benchmark_npu.sh \
  --warmup-runs 1 \
  --benchmark-runs 1 \
  --max-new-tokens 512
```

Raw front-camera resolution, closer to the actual planning path:

```bash
./local/run_planning_latency_benchmark_npu.sh \
  --scene-path /cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/openscene_meta_datas/086487f683be38e1b.pkl \
  --warmup-runs 0 \
  --benchmark-runs 1 \
  --max-new-tokens 512 \
  --use-raw-resolution
```

## Notes

- `64` or `256` new tokens were not enough for this CoT prompt and caused fallback to constant-velocity because the JSON output was truncated before the trajectory finished.
- `512` new tokens was enough to avoid fallback in the tested scene.
- The reported `agent_overhead_sec` is the time outside model forward, approximated as:
  `compute_trajectory total - local backend forward total`

## Observed Results

From the current machine on April 6, 2026:

- fixed `1280x704`, 1 real benchmark scene, `max-new-tokens=512`:
  about `346.97s`
- raw `1920x1080`, 1 real benchmark scene, `max-new-tokens=512`:
  about `611.01s`

Saved reports:

- `/cache/ma-user/curious_vla_assets/logs/planning_latency_benchmark_npu_20260406_100458.json`
- `/cache/ma-user/curious_vla_assets/logs/planning_latency_benchmark_npu_20260406_101232.json`
