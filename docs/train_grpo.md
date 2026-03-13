# How to Train Curious-VLA with GRPO?

Curious-VLA uses a RL training pipeline built on [EasyR1](../EasyR1/README.md):

Stages 2-3 form one **outer loop iteration**. Repeat for multiple rounds to progressively improve the policy.

> Throughout this guide, `$PROJECT_ROOT` refers to `/path/to/curious_vla`

## 0. Preparation

### Environment

Create the training environment in directory `curious_vla/EasyR1`:

```bash
conda create -n verl python=3.10
conda activate verl
cd EasyR1
pip install -e .
```

(Optional) Install flash-attention for faster training:

```bash
pip install flash-attn --no-build-isolation
```

More details: [EasyR1 README](../EasyR1/README.md)

### Model Weights (after SFT)

Place the SFT model under `EasyR1/checkpoints/sft/`:

```
EasyR1/checkpoints/sft/
└── your_model_name/
    ├── config.json
    ├── model-00001-of-00002.safetensors
    ├── ...
    └── tokenizer.json
```

**Coming soon.** Or train your own SFT model first.

### Training Data

The training data should be placed at `EasyR1/data/QA_navtrain_poutine_style_full` with the following structure:

```
EasyR1/data/QA_navtrain_poutine_style_full/
└── data/
    ├── train.parquet
    └── test.parquet
```

Additionally, link (or copy) your NAVSIM raw data into `EasyR1/navsim/` so that image paths in the parquet can be resolved:

```bash
ln -s /path/to/your/navsim_data $PROJECT_ROOT/EasyR1/navsim
```

Expected structure (symlink is fine):

```
EasyR1/navsim/
├── trainval_logs/trainval/
│   ├── 2021.05.12.19.36.12_veh-35_00005_00204.pkl
│   └── ...
└── trainval_sensor_blobs/trainval/
    ├── 2021.05.12.19.36.12_veh-35_00005_00204/
    │   └── CAM_F0/*.jpg
    └── ...
```

Each parquet file contains 3 columns:

| Column | Type | Description |
|--------|------|-------------|
| `images` | list[str] | Relative paths to front-view camera images (e.g., `navsim/trainval_sensor_blobs/.../*.jpg`) |
| `problem` | str | CoT prompt with `<image>` placeholder |
| `answer` | dict | `{"gt": [], "token": str}` — ground truth and scene token(gt is useless in RLVR) |

- **(Recommended)** Download pre-built parquet data:

  ```bash
  cd $PROJECT_ROOT/EasyR1/data
  huggingface-cli download MashiroLn/Curious-VLA --repo-type dataset --local-dir QA_navtrain_poutine_style_full
  ```

- **(Optional)** Build your own data from ShareGPT4V `.json`:

  ```bash
  cd $PROJECT_ROOT/EasyR1
  bash scripts/run_navsim_data.sh
  ```

### NAVSIM Reward Function API Server

The reward server scores trajectories during RL training. See [deploy.md](./deploy.md) and [navsim_eval/README.md](../navsim_eval/README.md) to prepare the `navsim` environment.

Steps:
- Build conda env
- Download data
- **Set env variables**
- **Build metric cache**:

## 1. Start Reward Function API Server

```bash
tmux new -s scorer
cd $PROJECT_ROOT/navsim_eval
bash gunicorn_server.sh
```

## 2. ADAS Inference & Filter

**Step 1**: Run parallel inference with the current model checkpoint, scoring each rollout with the NAVSIM reward server:

```bash
cd $PROJECT_ROOT
bash EasyR1/scripts/adas/run_adas_infer.sh
```

Key parameters in the script:
- `MODEL_PATH`: path to the current model checkpoint
- `data_path`: path to the training parquet data
- `worker.rollout.n`: number of rollouts per sample (e.g., 32)


**Step 2**: Filter dynamic samples.

```bash
cd $PROJECT_ROOT
bash EasyR1/scripts/adas/run_adas_filter.sh
```

Key parameters:
- `INFER_FOLDER`: path to the inference output from Step 1
- `-p`: score percentile threshold for filtering
- `--conf`: confidence threshold

This outputs a token filter file used in the next training step.

## 3. GRPO Training

Train the model with GRPO on the filtered data:

```bash
cd $PROJECT_ROOT/EasyR1
bash train_scripts/train_qwen_2_5_vl.sh
```

Refer to `EasyR1/examples/config_vla.yaml` for the full configuration.

## 4. Outer Loop

Repeat **Step 2 (ADAS) + Step 3 (GRPO)** with the updated checkpoint from each round.

Tips for resuming training across rounds:
- Delete `dataloader.pt` in the resume checkpoint directory if the data filter file has changed
- Reset the `epoch` to allocate more training steps; otherwise the trainer may send `SIGTERM`.
