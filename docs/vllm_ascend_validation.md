# vLLM Ascend Validation Notes

## Goal

This note records the practical adaptation path for running the local
`Curious-VLA` model with `vllm-ascend` on the current NPU machine, plus the
current validation status and remaining risks.

The goal of this round was not to finish the full benchmark integration yet.
The goal was to answer a narrower question first:

- Can `vllm-ascend` be installed in an isolated env on this machine?
- Can it load the local `Curious-VLA` model on NPU?
- Can it expose a working OpenAI-compatible API endpoint and return a response?

## Machine And Environment

- Host arch: `aarch64`
- OS: `EulerOS 2.0 (SP10)`
- NPU: `Ascend 910B3`
- CANN: `8.1.RC1`
- Working env path:
  `/cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend`
- Model path:
  `/cache/ma-user/curious_vla_assets/models/Curious-VLA`

Important context:

- The existing latency env
  `/cache/ma-user/curious_vla_assets/envs/curious-vla-npu-latency`
  was left untouched.
- All `vllm-ascend` work was done in a separate env.

## Why Standard Install Failed

The straightforward command:

```bash
pip install vllm==0.9.1
```

did not work directly on this machine for two separate reasons.

First, `vllm 0.9.1` does not provide an `aarch64` wheel on PyPI. Only these
artifacts were available:

- `vllm-0.9.1.tar.gz`
- `vllm-0.9.1-cp38-abi3-manylinux1_x86_64.whl`

So on this machine, `vllm` must be built from source or installed from a local
source tree.

Second, if built in the default way, the source build falls back to compiling
CPU custom ops, which originally failed with the system compiler:

- system `g++`: `7.3.0`
- build error:
  `invalid feature modifier in '-march=armv8.2-a+dotprod+fp16'`

## Working Install Strategy

The working strategy was to install:

1. `vllm-ascend==0.9.1`
2. upstream `vllm` from source, but with `VLLM_TARGET_DEVICE=empty`

The key idea is:

- `VLLM_TARGET_DEVICE=empty` skips upstream custom-op compilation
- `vllm-ascend` still provides the Ascend-specific plugin, kernels and patches
- this avoids the `aarch64` CPU custom-op build problem

### Installed Base Stack

Core versions in the final working env:

- `vllm`: `0.9.1+empty`
- `vllm-ascend`: `0.9.1`
- `torch`: `2.5.1`
- `torch_npu`: `2.5.1.post1`
- `transformers`: `4.52.4`
- `numpy`: `1.26.4`

### Practical Install Sequence

The successful install flow was:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate /cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend

pip install vllm-ascend==0.9.1
```

Then install upstream `vllm` from a local source tree:

```bash
pip download --no-deps vllm==0.9.1
tar -xf vllm-0.9.1.tar.gz
export VLLM_TARGET_DEVICE=empty
pip install --no-deps ./vllm-0.9.1
```

This installs `vllm` as `0.9.1+empty`.

Then install missing runtime dependencies required by `vllm` CLI and plugin
registration. In practice, the missing packages were discovered incrementally
while validating `import vllm` and `vllm --help`.

Main packages added:

- `pydantic`
- `fastapi[standard]`
- `openai`
- `aiohttp`
- `prometheus_client`
- `prometheus-fastapi-instrumentator`
- `tiktoken`
- `msgspec`
- `pyzmq`
- `sentencepiece`
- `protobuf`
- `psutil`
- `cachetools`
- `blake3`
- `py-cpuinfo`
- `gguf`
- `lm-format-enforcer`
- `outlines`
- `lark`
- `watchfiles`
- `python-json-logger`
- `opentelemetry-*`
- `compressed-tensors==0.10.1`
- `depyf==0.18.0`
- `llguidance`
- `xgrammar==0.1.19`
- `mistral_common[opencv]`
- `opencv-python-headless`

## Important Device Mapping Detail

This machine exposes only one visible NPU to the process, and it is indexed as
logical device `0`, even though `npu-smi` shows the physical card as `NPU 2`.

Practical consequence:

- `ASCEND_RT_VISIBLE_DEVICES=2` is wrong in this container
- `ASCEND_RT_VISIBLE_DEVICES=0` works
- leaving it unset also works

This was verified with:

```python
import torch
print(torch.npu.device_count())   # 1
print(torch.npu.current_device()) # 0
```

Setting `ASCEND_RT_VISIBLE_DEVICES=2` caused:

- `aclrtGetDeviceCount` failure
- `Set visible device failed, invalid device=0, input visible devices:2`

## What Was Successfully Validated

### 1. Import And CLI

These now work in the isolated env:

```bash
python -c "import vllm; import vllm_ascend"
vllm --help
```

Observed behavior:

- `vllm` loads successfully
- the Ascend platform plugin is discovered and activated
- Ascend model overrides for Qwen/Qwen-VL classes are registered

The warning below is expected in this install mode:

```text
Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
```

Reason:

- upstream `vllm` was intentionally installed as `VLLM_TARGET_DEVICE=empty`
- so upstream custom C++/CUDA ops are not built

At the moment, that warning does not block NPU service startup.

### 2. Model Load On NPU

The following command successfully loaded the local model and started the API:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate /cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend

export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /cache/ma-user/curious_vla_assets/models/Curious-VLA \
  --host 127.0.0.1 \
  --port 18000 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

Key log signals from the successful run:

- `Platform plugin ascend is activated`
- `Initializing a V0 LLM engine (v0.9.1)`
- `Starting to load model ...`
- `Loading weights took 3.79 seconds`
- `Loading model weights took 7.4149 GB`
- `init engine (profile, create kv cache, warmup model) took 22.50 seconds`
- `Starting vLLM API server 0 on http://127.0.0.1:18000`

### 3. API Reachability

`/v1/models` responded successfully:

```json
{
  "object": "list",
  "data": [
    {
      "id": "/cache/ma-user/curious_vla_assets/models/Curious-VLA",
      "object": "model"
    }
  ]
}
```

### 4. End-To-End Request / Response

A minimal text-only request to `/v1/chat/completions` returned HTTP 200 and a
completion payload.

This proves:

- process startup works
- engine startup works
- model loading works
- scheduler path works
- request routing works
- token generation path works

## Current Quality Assessment

Although the API returned a response, the content quality was not reasonable
for the prompt used.

Prompt:

```text
Say hello in one short sentence.
```

Observed response:

```text
{
  "no",
    "no",
    "no",
    "no
```

Interpretation:

- The stack is working at the system level.
- The current smoke test is not yet a semantic correctness test.
- The result is enough to say "vllm-ascend path can run end to end on this
  machine", but not enough to say "the model output is trustworthy in this
  serving mode".

## Why Output Sanity Is Still Weak

There are several reasons this should not yet be treated as a full correctness
validation:

- `Curious-VLA` is a vision-language model, while the smoke test used a pure
  text prompt.
- The test only checked that the server returns something, not that the answer
  matches an expected target.
- No image/video input was exercised yet.
- No dataset-backed evaluation was run through the `vllm-ascend` serving path.
- No PDM/planning-quality metric was tied to this serving backend yet.

So the current status is:

- system-level validation: yes
- semantic validation: not complete

## Known Warnings And Risks

### 1. Official Version Expectation vs Current Machine

The `vllm-ascend 0.9.1` package metadata mentions:

- `CANN >= 8.2.RC1`

This machine is:

- `CANN 8.1.RC1`

Despite that mismatch, the smoke test above still succeeded. So:

- current practical conclusion: usable enough to run a basic server
- remaining risk: there may still be feature gaps or latent incompatibilities
  under heavier workloads

### 2. `vllm._C` Not Built

This is intentional for the current installation method, but it also means:

- we are not using upstream compiled custom ops
- performance characteristics may differ from a more complete native build

### 3. OpenCV / NumPy Version Pressure

`opencv-python-headless` tried to pull a newer `numpy`, but the env was forced
back to `numpy 1.26.4` because `vllm-ascend 0.9.1` requires `numpy < 2.0.0`.

Current practical status:

- `cv2` imports successfully
- `vllm-ascend` imports successfully

But this is still a dependency edge worth keeping in mind.

## Current Conclusion

At this point, `vllm-ascend` on this machine has reached a meaningful
end-to-end milestone:

- isolated env created
- upstream `vllm` installed in a plugin-compatible form
- Ascend plugin loaded
- local `Curious-VLA` model loaded on NPU
- OpenAI-compatible API started
- `/v1/models` works
- `/v1/chat/completions` returns a response

So the answer to "can we bring up a `vllm-ascend` serving path for this
project on the current NPU machine?" is:

- yes, at smoke-test level

But the answer to "is this already a trustworthy benchmark backend?" is:

- not yet

## Recommended Next Steps

1. Add a small repeatable smoke-test script for the exact startup command and
   curl checks.
2. Design sanity-check prompts that match the model type:
   text-only prompts for text behavior, and image-conditioned prompts for VL
   behavior.
3. Add output validation rules before using this path for latency comparison.
4. Compare this `vllm-ascend` path with the existing transformers-based NPU
   benchmark path under the same prompt set.
5. If needed, bridge this back into the original LLaMA-Factory API flow, since
   the CUDA-side project path uses `LLaMA-Factory API + infer_backend=vllm`.

## 2026-04-07 Semantic Validation Update

On April 7, 2026, the validation scope was extended from "server smoke test"
to "Curious-VLA planning sanity gate".

This was done to answer a more practical question:

- before running latency numbers on `vllm-ascend`, can this backend return
  outputs that are at least structurally and semantically reasonable for the
  actual Curious-VLA planning contract?

### Validation Script

The validation entrypoints are now:

- `local/run_vllm_semantic_validation.py`
- `local/run_vllm_semantic_validation.sh`

The script does not try to run full NAVSIM evaluation. Instead, it checks
whether the OpenAI-compatible `vllm-ascend` endpoint can satisfy a smaller but
more relevant gate for Curious-VLA.

### Validation Suite Design

The suite now contains three layers.

1. Text-only schema control

- Purpose: detect whether the server can obey a strict JSON schema at all
- Input: no image, exact target schema
- Role: weak control only, not a deployment gate

2. Text-only planning-style control

- Purpose: test a planning-shaped prompt without vision input
- Input:
  - high-level intent = `go straight`
  - short past trajectory
  - explicit requirement for a smooth lane-following future trajectory
- Checks:
  - strict JSON parse
  - required keys present
  - `meta_behaviour.command` aligned with intent
  - denormalized trajectory sanity
  - forward-progress / lateral / heading sanity

3. Real VL planning samples

- Purpose: test the actual Curious-VLA usage mode
- Input:
  - real `warmup_two_stage` scene pickles
  - real `CAM_F0` front-view images
  - real planning prompt reconstructed from
    `NavsimCoTQwenAgent._build_prompt_messages()`
- Scene selection:
  - `diverse-by-command`
  - round-robin sampling across available warmup intents
  - current warmup split on this machine contains mostly `go straight` plus a
    smaller number of `turn left`
- Checks:
  - strict JSON contract
  - `critical_objects` completeness
  - `meta_behaviour` value validity
  - intent alignment between dataset command and predicted meta command
  - trajectory parseability
  - denormalized trajectory sanity
  - first-step XY error against the only currently available warmup future point

### Why Image Resize Was Added

Raw front-view images are `1920x1080`, and that was too expensive for the
current NPU `vllm-ascend` setup.

Processor-side prompt token estimates for the same real planning sample were:

- `1920x1080`: `3603`
- `1280x704`: `2062`
- `960x540`: `1558`
- `640x360`: `1211`

This matters because:

- `2048` context was enough to start the server
- but raw-resolution VL prompts were too long
- `4096` and `8192` startup attempts previously failed during vLLM profile /
  warmup with NPU OOM

So the practical compromise for semantic validation became:

- `vllm serve ... --max-model-len 2048`
- validation requests sent with resized images at `960x540`

### Practical Results On 2026-04-07

#### Text-only schema control

Result:

- request succeeded
- trajectory string was still parseable
- strict JSON contract was not valid

Interpretation:

- this backend is still weak on pure schema-copy behavior
- this check should not be used as the main latency gate

#### Text-only planning-style control

Result:

- request succeeded
- contract valid
- intent alignment valid
- trajectory sanity valid
- overall valid

This is important because it shows the backend can follow a planning-shaped
contract when the vision component is removed.

#### Real VL planning batch

Command class mix:

- `go straight`
- `turn left`

Batch result from:

- `/cache/ma-user/curious_vla_assets/logs/vllm_semantic_validation_20260407_batch6.json`

Summary:

- scene count: `6`
- request ok: `6/6`
- contract valid: `6/6`
- intent alignment valid: `6/6`
- trajectory valid: `6/6`
- overall valid: `6/6`
- mean latency: `12.979475s`
- p50 latency: `12.969514s`
- p95 latency: `13.290405s`

Observed details:

- straight scenes tended to produce `meta_behaviour.command = straight`
- left-turn scenes tended to produce `meta_behaviour.command = left_turn`
- first-step error against the single available future point stayed small in
  these samples

### Current Decision About Latency Benchmarking

Current decision:

- yes, this backend can now be used behind a lightweight semantic gate before
  latency benchmarking

But that statement is only true under the following conditions:

- use resized VL inputs, currently `960x540`
- keep the current working server shape around `max-model-len=2048`
- treat the gate as a "basic planning sanity gate", not as a full planning
  quality evaluation

What this decision does **not** mean:

- it does not prove full PDM / EPDMS correctness
- it does not validate raw `1920x1080` deployment behavior
- it does not replace full NAVSIM planner evaluation

### Recommended Use

For the current repo state, the safest workflow is:

1. Start `vllm-ascend` with the proven working configuration.
2. Run `local/run_vllm_semantic_validation.sh`.
3. Only if the VL batch passes, proceed to latency measurement for the same
   backend / image-size setting.

That makes the latency number more meaningful than a pure "server returned
something" measurement, while still staying much cheaper than full PDM
evaluation.
