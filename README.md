<div align="center">

# Curious-VLA

### Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models

[![arXiv](https://img.shields.io/badge/arXiv-2603.06049-b31b1b.svg)](https://arxiv.org/abs/2603.06049)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/MashiroLn/Curious-VLA)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![CVPR](https://img.shields.io/badge/CVPR%202026-Findings-blue.svg)]()

</div>

## News 

- 🎉 **CuriousVLA has been recommended to CVPR'26 Findings.** CuriousVLA is very suitable to serve as MLLM auto-regression-based SoTA baseline to compare. 

## TODOs

- [x] Paper release
- [x] Model weights on HuggingFace
- [x] Prompt Construction and Evaluation code
- [ ] Quick Start of environment(sft + rl + eval unified)(before Apr. 10)
- [ ] Training data (IL stage)(before Apr. 10)
- [ ] Model Weight after SFT (IL stage)(before Apr. 10)
- [x] Training code (RL stage: Basic reward function api server + SDR)
- [x] Training code (RL stage: ADAS)

Most will be released before late March, 2026. 

For obvious reasons about a certain confernce, our recent schedule has been significantly affected 🤷, and the remaining components will be fully released by April 10.

- [ ] Whole Data Engine
- [ ] Lightning-fast(15x throughput) reward function api server, and faster evaluation

Above will be released future.

## Overview

<div align="center">
<img src="assets/method_overview.png" width="90%">
</div>

## Abstract

Imitation Learning (IL) followed by Reinforcement Learning (RL) has emerged as a powerful paradigm for Vision-Language-Action (VLA) models in autonomous driving. However, we identify a critical bottleneck: IL training produces **overly narrow policies** that collapse exploration and limit the potential of subsequent RL stages, causing RL to saturate prematurely due to insufficient feedback diversity.

We propose **Curious-VLA**, a two-stage framework that unleashes exploration across both IL and RL stages through three key designs:

- **Feasible Trajectory Expansion (FTE):** Generates multiple physically valid trajectories with step-wise normalized representation during IL, broadening the behavioral diversity of the learned policy.
- **Adaptive Diversity-Aware Sampling (ADAS):** Prioritizes high-diversity samples during RL to maintain exploration breadth.
- **Spanning Driving Reward (SDR):** Uses focal-style weighting to amplify the reward's value span, improving sensitivity to differences in driving quality.


## Results

Performance on the **Navsim** benchmark:

| Method | PDMS | EPDMS(V2) | Best-of-N PDMS |
|:---:|:---:|:---:|:---:|
| **Curious-VLA** | **90.3** | **85.4** | **94.8** |

<!-- TODO: add more baselines for comparison -->

## Model Zoo

| Model | Base | Link |
|:---:|:---:|:---:|
| Curious-VLA | Qwen2.5-VL-3B-Instruct | [HuggingFace](https://huggingface.co/MashiroLn/Curious-VLA) |

## Getting Started

### Evaluation

You need to prepare **2 python venv**: `lf`(Llamafactory for deploy, py3.11), `navsim`(for evaluation, py3.9)

Details and Commands See [docs/deploy.md](docs/deploy.md)

### Training

#### 1. SFT: comming soon

#### 2. RL:

You need to prepare **2 python venv** for `EasyR1`, `navsim`.

Details and Commands See [docs/train_grpo.md](docs/train_grpo.md).
<!-- 

### Training

```bash
# TODO
``` -->

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{chen2026devilnarrowpolicyunleashing,
      title={Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models},
      author={Canyu Chen and Yuguang Yang and Zhewen Tan and Yizhi Wang and Ruiyi Zhan and Haiyan Liu and Xuanyao Mao and Jason Bao and Xinyue Tang and Linlin Yang and Bingchuan Sun and Yan Wang and Baochang Zhang},
      year={2026},
      eprint={2603.06049},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.06049},
}
```

## Acknowledgements

This project is built upon [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [Navsim](https://github.com/autonomousvision/navsim), [EasyR1](https://github.com/hiyouga/EasyR1.git), [VeRL](https://github.com/verl-project/verl.git). Code for Navsim Agent is inspired by [ReCogDrive](https://github.com/xiaomi-research/recogdrive.git). We thank the open-source community for their contributions.

## License

This project is released under the [Apache 2.0 License](LICENSE).
