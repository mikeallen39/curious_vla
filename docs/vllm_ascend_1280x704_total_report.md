# Curious-VLA vLLM-Ascend 1280x704 Total Report

This report summarizes the current practical status of running
`Curious-VLA` with `vllm-ascend` on the current Ascend NPU machine, with a
focus on the `1280x704` image setting that is closer to the project's earlier
CUDA-side benchmark setup.

Date of latest validation in this report:

- April 7, 2026

## 1. Scope

This report answers four concrete questions:

1. Can `vllm-ascend` serve the local `Curious-VLA` model on the current NPU?
2. Can it accept real `Curious-VLA` planning prompts with `1280x704` images?
3. Can it pass a lightweight semantic gate before latency benchmarking?
4. What latency numbers do we currently observe under this setting?

This report does **not** claim:

- full PDM / EPDMS correctness
- raw `1920x1080` deployment stability
- exact apples-to-apples parity with the in-process `transformers` planning
  benchmark

## 2. Machine And Assets

Machine:

- OS: `EulerOS 2.0 (SP10)`
- NPU: `Ascend 910B3`
- CANN: `8.1.RC1`
- arch: `aarch64`

Main asset paths:

- model:
  `/cache/ma-user/curious_vla_assets/models/Curious-VLA`
- warmup data:
  `/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage`
- vLLM env:
  `/cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend`
- logs:
  `/cache/ma-user/curious_vla_assets/logs`

## 3. Working vLLM-Ascend Configuration For 1280x704

The earlier `2048`-context server was enough for `960x540`, but not reliable
for `1280x704`.

The currently validated working configuration for `1280x704` is:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate /cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend

export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /cache/ma-user/curious_vla_assets/models/Curious-VLA \
  --host 127.0.0.1 \
  --port 18002 \
  --dtype bfloat16 \
  --max-model-len 2560 \
  --max-num-batched-tokens 2560 \
  --max-num-seqs 1 \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

Important note:

- `max_new_tokens=512` was **not** the blocker
- the real blocker was context capacity for the multimodal prompt
- for `1280x704`, the practical fix was raising `max-model-len` to `2560`
  while keeping scheduler settings conservative

## 4. Why 1280x704 Needs More Context Than 960x540

For the same real planning sample, processor-side prompt token estimates were:

- `1920x1080`: about `3603`
- `1280x704`: about `2062`
- `960x540`: about `1558`
- `640x360`: about `1211`

Implication:

- `960x540` fits comfortably under `2048`
- `1280x704` is already above that threshold
- raising context too aggressively on this machine can trigger NPU OOM during
  vLLM profile / warmup

The validated compromise was:

- `1280x704`
- `max-model-len=2560`
- `max-num-batched-tokens=2560`
- `max-num-seqs=1`

## 5. Semantic Gate Design

Before using this backend for latency measurement, a lightweight semantic gate
was added.

Main entrypoints:

- [run_vllm_semantic_validation.py](/home/ma-user/curious_vla/local/run_vllm_semantic_validation.py)
- [run_vllm_semantic_validation.sh](/home/ma-user/curious_vla/local/run_vllm_semantic_validation.sh)

The gate has three layers:

1. Text-only schema control
2. Text-only planning-style control
3. Real VL planning samples with real warmup scenes and real front-view images

The key checks are:

- strict JSON contract
- `critical_objects` completeness
- `meta_behaviour` value validity
- intent alignment between dataset command and predicted meta command
- trajectory parseability
- denormalized trajectory sanity
- first-step XY error against the single currently available warmup future point

## 6. Semantic Validation Results

### 6.1 960x540 batch gate

Report:

- `/cache/ma-user/curious_vla_assets/logs/vllm_semantic_validation_20260407_batch6.json`

Summary:

- scene count: `6`
- request ok: `6/6`
- overall valid: `6/6`
- mean latency: `12.979475s`

This established that the semantic gate itself was viable on `vllm-ascend`.

### 6.2 1280x704 gate

Report:

- `/cache/ma-user/curious_vla_assets/logs/vllm_semantic_validation_1280x704_20260407.json`

Validated on two real scenes:

- `go straight`
- `turn left`

Summary:

- request ok: `2/2`
- overall valid: `2/2`
- mean latency: `15.245859s`

This is the first direct confirmation that `1280x704` is workable on the
current machine under the `2560`-context server shape.

## 7. New Benchmark Entry

To avoid mixing backend semantics, a dedicated `vllm-ascend` planning latency
benchmark entry was added instead of reusing the in-process `transformers`
benchmark script.

New files:

- [local_env_vllm_ascend.sh](/home/ma-user/curious_vla/local/local_env_vllm_ascend.sh)
- [run_vllm_planning_latency_benchmark.py](/home/ma-user/curious_vla/local/run_vllm_planning_latency_benchmark.py)
- [run_vllm_planning_latency_benchmark.sh](/home/ma-user/curious_vla/local/run_vllm_planning_latency_benchmark.sh)
- [run_planning_latency_benchmark.sh](/home/ma-user/curious_vla/local/run_planning_latency_benchmark.sh)

What this benchmark measures per sample:

- full local scene-side handling in the benchmark client:
  - scene pickle load
  - prompt assembly
  - image resize
  - image base64 packaging
  - HTTP request
  - response parse
  - lightweight validation
- request latency itself
- client-side overhead outside the request

This is still **not** the same metric as:

- `NavsimCoTQwenAgent.compute_trajectory()` local in-process benchmark on
  `transformers + torch_npu`

So the numbers below are useful, but should not be compared naively against the
older local-agent benchmark.

## 8. Formal 1280x704 Benchmark Result

Benchmark report:

- `/cache/ma-user/curious_vla_assets/logs/vllm_planning_latency_benchmark_20260407_1280x704.json`

Run setup:

- base URL: `http://127.0.0.1:18002/v1`
- image size: `1280x704`
- warmup runs: `1`
- benchmark runs: `4`
- scene selection: `diverse-by-command`
- text planning gate: enabled

Scene mix:

- `go straight`
- `turn left`

### 8.1 Gate Result

- text planning control: passed

### 8.2 Benchmark Summary

- benchmark count: `4`
- request ok: `4/4`
- contract valid: `4/4`
- intent alignment valid: `4/4`
- trajectory valid: `4/4`
- overall valid: `4/4`
- recommended for latency benchmark: `true`

Latency summary:

- mean request latency: `13.188920s`
- p50 request latency: `13.200862s`
- p95 request latency: `13.371148s`
- min request latency: `12.956358s`
- max request latency: `13.397599s`

End-to-end scene-side summary in the benchmark client:

- mean total scene time: `13.262334s`
- p50 total scene time: `13.271470s`
- p95 total scene time: `13.450370s`

Client overhead outside the HTTP request:

- mean overhead: `0.073414s`

Interpretation:

- most of the measured time is in the server-side completion path
- benchmark-client overhead is currently small relative to request latency

## 9. Relation To The Existing Transformers Benchmark

The older in-process planning benchmark on `transformers + torch_npu` recorded:

- fixed `1280x704`, 1 real benchmark scene:
  about `346.97s`

That path measures something different:

- local model forward
- local agent path
- local processor / tokenizer path
- `NavsimCoTQwenAgent.compute_trajectory()`

The new `vllm-ascend` benchmark instead measures:

- client-side scene preparation
- OpenAI-compatible server request / response latency
- lightweight semantic validation

So:

- these two benchmarks are related
- but they are not yet directly comparable as if they were the same metric

The current `vllm-ascend` result is best interpreted as:

- server-style planning response latency after a semantic gate

not as:

- full local planner latency replacement

## 10. Backend Switching

There is now a unified benchmark entry:

- [run_planning_latency_benchmark.sh](/home/ma-user/curious_vla/local/run_planning_latency_benchmark.sh)

It dispatches by backend:

- `--backend transformers`
  - calls the existing in-process NPU benchmark
  - underlying script:
    [run_planning_latency_benchmark_npu.sh](/home/ma-user/curious_vla/local/run_planning_latency_benchmark_npu.sh)
- `--backend vllm`
  - calls the new server-style `vllm-ascend` benchmark
  - underlying script:
    [run_vllm_planning_latency_benchmark.sh](/home/ma-user/curious_vla/local/run_vllm_planning_latency_benchmark.sh)

So the practical switch point is now one command-line argument, not a manual
script-name swap.

## 11. Current Practical Conclusion

Current conclusion on April 7, 2026:

- `vllm-ascend` can now run `Curious-VLA` with `1280x704` images on this NPU
- it can pass a lightweight planning-oriented semantic gate
- it can be used for a server-style latency benchmark under the proven config

The currently recommended practical workflow is:

1. Start `vllm-ascend` with the validated `2560`-context config.
2. Run semantic validation first.
3. If the gate passes, run the `vllm` planning latency benchmark.

## 12. Remaining Gaps

The following are still open:

- no full PDM / EPDMS quality evaluation through the `vllm-ascend` backend
- no raw `1920x1080` stable benchmark path confirmed for this backend
- no larger multi-scene stress run yet for `1280x704`
- no strict apples-to-apples benchmark tying the server path back to
  `SceneLoader + sequential worker + full NAVSIM evaluation`

## 13. Recommended Commands

Start service:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate /cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend

export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /cache/ma-user/curious_vla_assets/models/Curious-VLA \
  --host 127.0.0.1 \
  --port 18002 \
  --dtype bfloat16 \
  --max-model-len 2560 \
  --max-num-batched-tokens 2560 \
  --max-num-seqs 1 \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

Run semantic gate:

```bash
BASE_URL=http://127.0.0.1:18002/v1 \
WIDTH=1280 \
HEIGHT=704 \
./local/run_vllm_semantic_validation.sh \
  --scene-limit 2 \
  --selection-mode diverse-by-command
```

Run benchmark:

```bash
BASE_URL=http://127.0.0.1:18002/v1 \
WIDTH=1280 \
HEIGHT=704 \
./local/run_planning_latency_benchmark.sh \
  --backend vllm \
  --scene-limit 4 \
  --warmup-runs 1 \
  --benchmark-runs 4
```

Run the old in-process transformers benchmark through the same unified entry:

```bash
./local/run_planning_latency_benchmark.sh \
  --backend transformers \
  --warmup-runs 1 \
  --benchmark-runs 1 \
  --max-new-tokens 512
```
