# How to Train Curious-VLA with GRPO?

Curious-VLA uses a RL training pipeline built on [EasyR1](../EasyR1/README.md). Use `curious` and `navsim` conda envs (created by `bash scripts/setup_env.sh`).

There are two ways to start GRPO:

- Recommended: use the released SFT checkpoint `curious_vla_qwen2_5_vl_3b_sft_stage2` together with the released filter-token file, and start GRPO directly
- Optional: rerun ADAS yourself, then use your own ADAS output as `ADAS_FILTER_FILE`

## Data Preparation

### Model Weights (after SFT)

Place the SFT model under `EasyR1/checkpoints/sft/`, or train your own (see [train_sft.md](train_sft.md)).

### Training Data

**(Recommended)** Download pre-built parquet data:

```bash
cd /path/to/curious_vla/EasyR1/data
huggingface-cli download MashiroLn/Curious-VLA --repo-type dataset --local-dir QA_navtrain_poutine_style_full
```

Expected layout after download:

```text
EasyR1/data/QA_navtrain_poutine_style_full/
└── data/
    ├── train.parquet
    └── test.parquet
```

**(Optional)** Build your own data:

```bash
cd /path/to/curious_vla/EasyR1
bash scripts/run_navsim_data.sh
```

Link NAVSIM raw data so that image paths in the parquet can be resolved. Reuse the same repo-local data link described in [train_sft.md](train_sft.md):

```bash
cd /path/to/curious_vla
ln -s /path/to/navsim_workspace/dataset datasets/navsim
ln -s ../datasets/navsim EasyR1/navsim
```

### Reward Server

The reward server scores trajectories during RL training. It is **automatically started** by `scripts/run_adas.sh` and `scripts/run_rl_train.sh`.

Before first use, build the `navtrain` metric cache:

```bash
DATA_ROOT="$PROJECT_ROOT/datasets/navsim"
TRAIN_TEST_SPLIT="navtrain"
```

Then run:

```bash
bash navsim_eval/scripts/evaluation/run_metric_caching.sh
```

## ADAS Inference & Filter

This section is optional. Only run it if you want to rebuild the ADAS filter yourself.

Edit `scripts/run_adas.sh` and fill in:

- `MODEL_PATH`
- `EXP_NAME`
- `NUM_GPUS` if needed

Keep `DATA_ROOT="$PROJECT_ROOT/datasets/navsim"` unless you also change the repo-local data link.

Run ADAS with the SFT checkpoint:

```bash
bash scripts/run_adas.sh
```

This starts the reward server, runs ADAS inference and filtering, and outputs a token filter file at `EasyR1/checkpoints/adas/<exp_name>/group_stats_filtered_0.1.txt`.

Released token-filter file for direct GRPO training:

```text
token_filters/curious_vla_qwen2_5_vl_3b_sft_stage2_adas1x_6k.txt
```

This file was recovered from the early filtered training subset `navsim_normtrajtext_cot_filter_dynamic_6k/data/train.parquet`. The format is one token per line.

## GRPO Training

Edit `scripts/run_rl_train.sh` and fill in:

- `MODEL_PATH`
- `EXP_NAME`
- `NUM_GPUS` if needed

Recommended:

- use the released SFT checkpoint `curious_vla_qwen2_5_vl_3b_sft_stage2`
- keep the default `ADAS_FILTER_FILE` in `scripts/run_rl_train.sh`
- run GRPO directly

Optional:

- rerun ADAS first
- replace `ADAS_FILTER_FILE` with your own ADAS output

```bash
bash scripts/run_rl_train.sh
```

Refer to `EasyR1/examples/config_vla.yaml` for the full configuration.

## Outer Loop

Repeat **ADAS + GRPO** with the updated checkpoint from each round.

Tips for resuming:
- Uncomment `RESUME_FROM` in both scripts and point it to the previous checkpoint
- Delete `dataloader.pt` in the resume checkpoint directory if the data filter file has changed
- Reset the `epoch` to allocate more training steps; otherwise the trainer may send `SIGTERM`

## Next Steps

The GRPO model can be directly evaluated. See [deploy.md](deploy.md).
