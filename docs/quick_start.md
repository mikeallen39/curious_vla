# Quick Start

## Environment Setup

Curious-VLA uses **2 conda environments**:

| Environment | Purpose |
|:---:|:---|
| `curious` | SFT training, RL training, model serving |
| `navsim` | Evaluation, reward function server |

```bash
bash scripts/setup_env.sh
```

`scripts/setup_env.sh` installs the in-repo `EasyR1` package, clones LLaMA-Factory into `LLaMA-Factory/`, installs it into the same `curious` env, and creates the `navsim` eval env.

(Required system tools for the entry scripts: `git`, `tmux`, `lsof`)

(Optional) Flash Attention:

```bash
conda activate curious && pip install flash-attn --no-build-isolation
```

---

## Data Preparation

### 1. NAVSIM Official Setup

Follow [navsim_eval/docs/install.md](../navsim_eval/docs/install.md) for:

- dataset download
- expected dataset directory layout
- NAVSIM environment variables

Then:

- see [deploy.md](deploy.md) to link eval-time test data
- see [train_sft.md](train_sft.md) to link SFT-time train data

### 2. Build Metric Cache (required before first eval)

First create the repo-local NAVSIM link used by all entry scripts:

```bash
cd /path/to/curious_vla
ln -s /path/to/navsim_workspace/dataset datasets/navsim
```

`DATA_ROOT` in the entry scripts is fixed to `"$PROJECT_ROOT/datasets/navsim"`, so this link is the single source of truth.

Evaluation uses the `navtest` cache. Edit `navsim_eval/scripts/evaluation/run_metric_caching.sh` so:

```bash
DATA_ROOT="$PROJECT_ROOT/datasets/navsim"
TRAIN_TEST_SPLIT="navtest"
```

Then run:

```bash
bash navsim_eval/scripts/evaluation/run_metric_caching.sh
```

RL reward server uses the `navtrain` cache. In the same script, change:

```bash
TRAIN_TEST_SPLIT="navtrain"
```

Then run:

```bash
bash navsim_eval/scripts/evaluation/run_metric_caching.sh
```

### 3. SFT Training Data

See [train_sft.md](train_sft.md).

### 4. Model Weights (for eval)

```bash
huggingface-cli download MashiroLn/Curious-VLA --local-dir /path/to/model
```

---

## Evaluation

Edit `scripts/run_eval.sh` and fill in at least:

- `MODEL_NAME_OR_PATH`
- `EXPERIMENT_NAME`
- `NUM_GPUS` if needed

Then run:

```bash
bash scripts/run_eval.sh
```

---

## SFT Training

Edit `sft/sft_stage1.yaml` and `sft/sft_stage2.yaml` (model path, output dir, etc.), then run:

```bash
bash scripts/run_sft_train.sh
```

This runs Stage 1 (warmup) and Stage 2 sequentially. See [train_sft.md](train_sft.md) for data preparation.

---

## RL Training (GRPO)

There are two ways to start GRPO. See [train_grpo.md](train_grpo.md) for details.

### Recommended: Direct GRPO from the released SFT checkpoint

If you use the released SFT checkpoint `curious_vla_qwen2_5_vl_3b_sft_stage2`, you can keep the default `ADAS_FILTER_FILE` in `scripts/run_rl_train.sh` and start GRPO directly.

Edit `scripts/run_rl_train.sh` and fill in:

- `MODEL_PATH`
- `EXP_NAME`
- `NUM_GPUS` if needed

Then run:

```bash
bash scripts/run_rl_train.sh
```

### Optional: Full RL loop with ADAS

### Step 1: ADAS

Edit `scripts/run_adas.sh` and fill in:

- `MODEL_PATH`
- `EXP_NAME`
- `NUM_GPUS` if needed

Keep `DATA_ROOT="$PROJECT_ROOT/datasets/navsim"` unless you also change the repo-local data link.

Run ADAS after building the `navtrain` metric cache:

```bash
bash scripts/run_adas.sh
```

### Step 2: GRPO Training

Edit `scripts/run_rl_train.sh` and fill in:

- `MODEL_PATH`
- `EXP_NAME`
- `ADAS_FILTER_FILE`
- `NUM_GPUS` if needed

Use the ADAS filter file from Step 1:

```bash
bash scripts/run_rl_train.sh
```

### Outer Loop

Repeat Step 1 and Step 2 with the updated checkpoint. Uncomment `RESUME_FROM` in both scripts to point to the previous round's checkpoint (for example `/path/to/checkpoint/global_step_x`). Delete `dataloader.pt` in the resume directory if the ADAS filter file has changed.
