# How to Deploy And Eval Curious-VLA?

Curious-VLA model is based on Qwen2.5-VL-3B, mainly focusing on the Training Paradigm.

The **Deployment** of Curious-VLA is the same as Qwen2.5-VL-3B.

## Quick Way

`bash scripts/setup_env.sh` already clones and installs LLaMA-Factory into the `curious` env.
First, follow [navsim_eval/docs/install.md](../navsim_eval/docs/install.md) for NAVSIM dataset download, directory layout, and environment variables.

Create the repo-local NAVSIM link used by all entry scripts:

```bash
cd /path/to/curious_vla
ln -s /path/to/navsim_workspace/dataset datasets/navsim
```

For evaluation, `datasets/navsim` should contain at least:

- `maps/`
- `navsim_logs/test/`
- `sensor_blobs/test/`

Build the `navtest` metric cache once. In `navsim_eval/scripts/evaluation/run_metric_caching.sh`, keep:

```bash
DATA_ROOT="$PROJECT_ROOT/datasets/navsim"
TRAIN_TEST_SPLIT="navtest"
```

Then run:

```bash
bash navsim_eval/scripts/evaluation/run_metric_caching.sh
```

Then edit `scripts/run_eval.sh` and fill in:

- `MODEL_NAME_OR_PATH`
- `EXPERIMENT_NAME`
- `NUM_GPUS` if needed

Then run the one-shot evaluation entry script:

```bash
bash scripts/run_eval.sh
```

This automatically handles serving, evaluation, and cleanup. The entry scripts require `git`, `tmux`, and `lsof`. See below for manual steps if needed.

## Manual Steps

### 1. Deploy (Server)

Run the deploy script in `navsim_eval/`:

```bash
cd navsim_eval/
model_name_or_path=/path/to/model TOTAL_GPUS=8 bash lf_serve_cot.sh 8
```

We recommend using LLaMA-Factory (`llamafactory-cli api`) rather than pure vLLM (`vllm serve`).
Although vLLM can also be used for deployment, we observed slightly lower performance compared to LLaMA-Factory, likely due to differences in chat template handling.

If you want to use `vllm serve`, see `navsim_eval/vllm_serve.sh`.

### 2. Prompt & Eval (Client)

Steps:
- Link the official NAVSIM dataset root to `datasets/navsim` as shown above
- Build metric cache (first time only) by editing `navsim_eval/scripts/evaluation/run_metric_caching.sh`:

```bash
DATA_ROOT="$PROJECT_ROOT/datasets/navsim"
TRAIN_TEST_SPLIT="navtest"
```

Then run:

```bash
bash navsim_eval/scripts/evaluation/run_metric_caching.sh
```

- Run evaluation by editing `navsim_eval/scripts/evaluation/run_qwen_pdm_score_evaluation.sh`:

```bash
DATA_ROOT="$PROJECT_ROOT/datasets/navsim"
EXPERIMENT_NAME="curious_vla_eval"
MODEL_NAME_OR_PATH="none"
```

Then run:

```bash
bash navsim_eval/scripts/evaluation/run_qwen_pdm_score_evaluation.sh
```

If you use `vllm serve` instead of the LF API server, set `MODEL_NAME_OR_PATH` to the actual model path.

### 3. Prompt Construction

See `navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py`
