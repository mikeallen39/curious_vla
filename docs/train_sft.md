# How to Train Curious-VLA with SFT?

Curious-VLA uses a two-stage SFT pipeline built on [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory). Use the `curious` conda env created by `bash scripts/setup_env.sh`. That setup script already clones and installs LLaMA-Factory into the env.

## Environment

```bash
bash scripts/setup_env.sh
```

## Data Preparation

First, follow [navsim_eval/docs/install.md](../navsim_eval/docs/install.md) to download the official NAVSIM data and set the expected environment variables.

Then download SFT annotations from [HuggingFace Repo](https://huggingface.co/MashiroLn/Curious-VLA-dev):

```bash
# SFT annotation data
huggingface-cli download MashiroLn/Curious-VLA-dev --repo-type dataset --local-dir /path/to/Curious-VLA-dev
```

Link data into the `datasets/` directory. The NAVSIM link should point to the official dataset root prepared by `navsim_eval/docs/install.md`:

```bash
cd /path/to/curious_vla

# Link SFT annotation json
ln -s /path/to/Curious-VLA-dev/CuriousVLA_data/QA_sft_navsim_train_cot_1view_103k_baseline_norm.json \
      datasets/QA_sft_navsim_train_cot_1view_103k_baseline_norm.json

# Link official NAVSIM dataset root (must contain trainval data)
ln -s /path/to/navsim_workspace/dataset datasets/navsim
```

`datasets/navsim` is also the default `DATA_ROOT` used by the eval and RL entry scripts.

Expected directory structure after linking:

```
datasets/
├── datasets_info.json                         # already in repo
├── QA_sft_navsim_train_cot_1view_103k_baseline_norm.json -> ...        # symlink
└── navsim -> /path/to/navsim_workspace/dataset/   # symlink
    ├── trainval_logs/trainval/*.pkl
    └── trainval_sensor_blobs/trainval/<scene_id>/CAM_F0/*.jpg
```

Multiple SFT data variants are available in `CuriousVLA_data/`. Switch by updating the `file_name` field in `datasets/datasets_info.json`. Recommended combinations:

1. baseline data + Qwen2.5-VL ✅
2. golden data + Qwen3-VL ✅
3. golden data + Qwen2.5-VL ❌

## Training

Edit `sft/sft_stage1.yaml` and `sft/sft_stage2.yaml` (model path, output dir, etc.), then:

```bash
bash scripts/run_sft_train.sh
```

This runs Stage 1 (warmup) and Stage 2 sequentially. The final SFT model will be saved to the `output_dir` specified in `sft/sft_stage2.yaml`.

## Next Steps

The SFT model can be directly evaluated. See [deploy.md](deploy.md).

To further improve performance with RL, see [train_grpo.md](train_grpo.md).
