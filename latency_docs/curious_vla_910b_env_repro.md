# Curious-VLA 在 Ascend 910B 上的环境复现文档

本文整理当前 `/home/ma-user/curious_vla` 跑通 NPU latency 相关实验时，实际使用的环境配置、安装顺序和校验方法，目标是方便后续在其它 `Ascend 910B` 机器上复现。

本文覆盖两套环境：

- 本地进程内 `transformers + torch_npu` 环境
- 服务化 `vllm-ascend` 环境

之所以分成两套 env，是因为：

- `transformers + torch_npu` 这条链更接近本地 agent 内推理
- `vllm-ascend` 这条链依赖另一组 `torch / torch_npu / transformers` 组合
- 两条链混装后容易相互污染

## 1. 当前成功机器的基线

当前已经验证通过的机器基线如下：

- OS：`EulerOS 2.0 (SP10)`
- 架构：`aarch64`
- NPU：`Ascend 910B3`
- CANN：`8.1.RC1`

如果你换到别的 `910B` 机器，最先确认的不是仓库代码，而是下面三件事：

1. `Ascend driver + CANN` 是否已正确安装
2. `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 后 `npu-smi info` 是否正常
3. 该机器能否拿到与本机兼容的 `torch_npu` wheel

## 2. 推荐目录布局

当前这套复现使用如下目录：

```bash
/home/ma-user/curious_vla
/cache/ma-user/curious_vla_assets/
  data/
    downloads/
      maps/
      test_navsim_logs/
      warmup_two_stage/
  envs/
    curious-vla-npu-latency/
    curious-vla-vllm-ascend/
  logs/
  models/
    Curious-VLA/
```

建议沿用同样的思路：

- 代码放在较小但稳定的目录
- 模型、数据、conda env 放在大盘

初始化目录：

```bash
mkdir -p /cache/ma-user/curious_vla_assets/{data/downloads,envs,logs,models}
```

## 3. 模型与数据下载

### 3.1 Hugging Face 镜像

建议统一使用 HF 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### 3.2 模型下载

当前实际使用模型：

- 模型仓库：`MashiroLn/Curious-VLA`
- 本地目录：`/cache/ma-user/curious_vla_assets/models/Curious-VLA`
- 模型基座：`Qwen2.5-VL-3B-Instruct`

下载命令：

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="MashiroLn/Curious-VLA", local_dir="/cache/ma-user/curious_vla_assets/models/Curious-VLA", max_workers=1)'
```

### 3.3 最小可用数据

当前为了跑通 latency benchmark，最小建议下载三类数据：

- `maps`
- `warmup_two_stage`
- `test metadata`

下载命令：

```bash
mkdir -p /cache/ma-user/curious_vla_assets/data/downloads
cd /cache/ma-user/curious_vla_assets/data/downloads

wget -c https://hf-mirror.com/datasets/pengxiang/nuplan_maps/resolve/main/nuplan-maps-v1.1.zip
unzip -o nuplan-maps-v1.1.zip
rm -f nuplan-maps-v1.1.zip
mv -f nuplan-maps-v1.0 maps

wget -c https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/navsim_v2.2_warmup_two_stage.tar.gz
tar -xzvf navsim_v2.2_warmup_two_stage.tar.gz
rm -f navsim_v2.2_warmup_two_stage.tar.gz

wget -c https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz
tar -xzf openscene_metadata_test.tgz
rm -f openscene_metadata_test.tgz
mv -f openscene-v1.1/meta_datas test_navsim_logs
rm -rf openscene-v1.1
```

### 3.4 兼容仓库访问方式的软链接

```bash
mkdir -p /cache/ma-user/curious_vla_assets/data/navsim_logs

ln -sfn /cache/ma-user/curious_vla_assets/data/downloads/maps \
  /cache/ma-user/curious_vla_assets/data/maps

ln -sfn /cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage \
  /cache/ma-user/curious_vla_assets/data/warmup_two_stage

ln -sfn /cache/ma-user/curious_vla_assets/data/downloads/test_navsim_logs/test \
  /cache/ma-user/curious_vla_assets/data/navsim_logs/test
```

注意最后一个软链接必须指向：

- `.../downloads/test_navsim_logs/test`

不能只指向：

- `.../downloads/test_navsim_logs`

## 4. 环境变量

推荐把下面两段分别保存成你自己的局部脚本，或直接沿用仓库里的：

- [local/local_env_npu.sh](/home/ma-user/curious_vla/local/local_env_npu.sh)
- [local/local_env_vllm_ascend.sh](/home/ma-user/curious_vla/local/local_env_vllm_ascend.sh)

本地 `transformers` 环境变量：

```bash
export PROJECT_ROOT="/home/ma-user/curious_vla"
export ASSET_ROOT="/cache/ma-user/curious_vla_assets"
export CONDA_ROOT="/home/ma-user/anaconda3"
export LATENCY_ENV_PREFIX="$ASSET_ROOT/envs/curious-vla-npu-latency"
export CURIOUS_VLA_MODEL_DIR="$ASSET_ROOT/models/Curious-VLA"
export WARMUP_ROOT="$ASSET_ROOT/data/downloads/warmup_two_stage"
export LATENCY_LOG_DIR="$ASSET_ROOT/logs"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT/navsim_eval"
export OPENSCENE_DATA_ROOT="$WARMUP_ROOT"
export WARMUP_SENSOR_PATH="$WARMUP_ROOT/sensor_blobs"
export WARMUP_SYNTHETIC_SCENES_PATH="$WARMUP_ROOT/synthetic_scene_pickles"
export STATS_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
```

`vllm-ascend` 环境变量：

```bash
export PROJECT_ROOT="/home/ma-user/curious_vla"
export ASSET_ROOT="/cache/ma-user/curious_vla_assets"
export CONDA_ROOT="/home/ma-user/anaconda3"
export VLLM_ASCEND_ENV_PREFIX="$ASSET_ROOT/envs/curious-vla-vllm-ascend"
export CURIOUS_VLA_MODEL_DIR="$ASSET_ROOT/models/Curious-VLA"
export WARMUP_ROOT="$ASSET_ROOT/data/downloads/warmup_two_stage"
export LATENCY_LOG_DIR="$ASSET_ROOT/logs"
export VLLM_ASCEND_BASE_URL="http://127.0.0.1:18002/v1"
export STATS_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
```

## 5. 环境一：`transformers + torch_npu`

### 5.1 当前验证通过的精确版本

当前实际可工作 conda prefix：

- `/cache/ma-user/curious_vla_assets/envs/curious-vla-npu-latency`

关键版本：

- `python==3.10.6`
- `torch==2.1.0`
- `torch-npu==2.1.0.post12`
- `torchvision==0.16.0`
- `transformers==4.55.4`
- `numpy==1.26.4`

其他关键包：

- `accelerate==1.0.1`
- `openai==2.30.0`
- `opencv-python==4.9.0.80`
- `opencv-python-headless==4.8.1.78`
- `pillow==10.4.0`
- `protobuf==4.25.7`
- `sentencepiece==0.2.0`
- `safetensors==0.4.5`
- `tokenizers==0.21.4`
- `einops==0.8.0`
- `peft==0.7.1`
- `transformers-stream-generator==0.0.5`

`navsim_eval` 相关包当前也已经导入通过，包括：

- `hydra-core==1.2.0`
- `pytorch-lightning==2.2.1`
- `geopandas==0.14.4`
- `shapely==2.0.6`
- `pyproj==3.7.1`
- `pyogrio==0.12.1`
- `rasterio==1.4.4`
- `aioboto3==15.5.0`

### 5.2 最接近当前机器的复现方式

本机当时是直接从已有 Ascend 基础环境克隆出来的：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh

conda create -y -p /cache/ma-user/curious_vla_assets/envs/curious-vla-npu-latency \
  --clone /home/ma-user/anaconda3/envs/PyTorch-2.1.0
```

如果你的目标是“尽量和当前机器保持一致”，这是最稳的做法。

### 5.3 在新 910B 机器上的推荐安装顺序

如果目标机器没有现成的 `PyTorch-2.1.0` 基础 env，推荐按下面顺序重建：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh

conda create -y -p /cache/ma-user/curious_vla_assets/envs/curious-vla-npu-latency python=3.10 pip
conda activate /cache/ma-user/curious_vla_assets/envs/curious-vla-npu-latency
```

第一步，先装 `torch / torchvision / torch-npu` 这一组兼容组合。

当前这台机器最终工作的是：

- `torch==2.1.0`
- `torchvision==0.16.0`
- `torch-npu==2.1.0.post12`

其中当前机器上的 `torch-npu` 来自内部 wheel，记录到的源信息是：

- `torch_npu-2.1.0.post12-cp310-...-aarch64.whl`

在其它机器上通常不能直接复用这个内网地址，所以你需要换成你自己环境里可访问的：

- Ascend 官方/镜像 wheel
- 或运维侧已经准备好的 wheel 仓库

第二步，安装 `navsim_eval` 依赖并做 editable install：

```bash
cd /home/ma-user/curious_vla/navsim_eval
pip install -r requirements.txt
pip install -e .
```

第三步，覆盖到当前验证通过的关键多模态版本：

```bash
pip install --upgrade --force-reinstall \
  numpy==1.26.4 \
  transformers==4.55.4 \
  accelerate==1.0.1 \
  openai==2.30.0 \
  opencv-python==4.9.0.80 \
  opencv-python-headless==4.8.1.78 \
  pillow==10.4.0 \
  protobuf==4.25.7 \
  safetensors==0.4.5 \
  sentencepiece==0.2.0 \
  tokenizers==0.21.4 \
  einops==0.8.0 \
  peft==0.7.1 \
  transformers-stream-generator==0.0.5
```

### 5.4 校验命令

基础校验：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
source /home/ma-user/curious_vla/local/local_env_npu.sh
conda activate "$LATENCY_ENV_PREFIX"

python - <<'PY'
import torch
import torch_npu
import transformers
import navsim
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("transformers:", transformers.__version__)
print("navsim:", navsim.__version__)
print("npu available:", torch.npu.is_available())
PY
```

进一步校验 agent 导入：

```bash
python - <<'PY'
from navsim.agents.curious_vla.navsim_qwen_norm_agent_cot import NavsimCoTQwenAgent
print("NavsimCoTQwenAgent import ok")
PY
```

## 6. 环境二：`vllm-ascend`

### 6.1 当前验证通过的精确版本

当前实际可工作 conda prefix：

- `/cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend`

关键版本：

- `python==3.10.6`
- `torch==2.5.1`
- `torch-npu==2.5.1.post1`
- `torchvision==0.20.1`
- `transformers==4.52.4`
- `numpy==1.26.4`
- `vllm-ascend==0.9.1`
- `vllm==0.9.1+empty`

其他关键包：

- `einops==0.8.2`
- `fastapi==0.135.3`
- `msgspec==0.20.0`
- `openai==2.30.0`
- `opencv-python-headless==4.13.0.92`
- `outlines==0.1.11`
- `pillow==12.2.0`
- `protobuf==6.33.6`
- `sentencepiece==0.2.1`
- `tokenizers==0.21.4`
- `uvicorn==0.44.0`

### 6.2 为什么不能直接 `pip install vllm==0.9.1`

在当前 `aarch64 + Ascend` 机器上，直接装 upstream `vllm==0.9.1` 不稳，主要原因是：

- PyPI 没有现成可用的 `aarch64` wheel
- 直接从源码装会走 upstream CPU custom ops 编译
- 当前系统编译链会在 `-march=armv8.2-a+dotprod+fp16` 这类参数上失败

所以这里不能按普通 CUDA 机器的方式装。

### 6.3 推荐安装顺序

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh

conda create -y -p /cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend python=3.10 pip
conda activate /cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend
```

先装当前验证通过的基础组合：

```bash
pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torch-npu==2.5.1.post1 \
  numpy==1.26.4 \
  transformers==4.52.4 \
  tokenizers==0.21.4 \
  sentencepiece==0.2.1 \
  pillow==12.2.0 \
  protobuf==6.33.6 \
  openai==2.30.0 \
  opencv-python-headless==4.13.0.92 \
  einops==0.8.2 \
  fastapi==0.135.3 \
  uvicorn==0.44.0 \
  msgspec==0.20.0 \
  outlines==0.1.11
```

再装 Ascend 插件：

```bash
pip install vllm-ascend==0.9.1
```

最后安装 upstream `vllm`，但必须跳过目标设备编译：

```bash
pip download --no-deps vllm==0.9.1
tar -xf vllm-0.9.1.tar.gz
export VLLM_TARGET_DEVICE=empty
pip install --no-deps ./vllm-0.9.1
```

这样最终得到的就是当前机器上已经验证通过的：

- `vllm==0.9.1+empty`
- `vllm-ascend==0.9.1`

### 6.4 校验命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
source /home/ma-user/curious_vla/local/local_env_vllm_ascend.sh
conda activate "$VLLM_ASCEND_ENV_PREFIX"

python -c "import vllm; import vllm_ascend; print(vllm.__version__)"
vllm --help
```

如果上面能过，继续拉起服务：

```bash
vllm serve /cache/ma-user/curious_vla_assets/models/Curious-VLA \
  --host 127.0.0.1 \
  --port 18002 \
  --dtype bfloat16 \
  --max-model-len 2560 \
  --max-num-seqs 1 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.6 \
  --trust-remote-code
```

## 7. 设备号注意事项

当前这台机器虽然在 `npu-smi` 里显示物理卡号不一定是 `0`，但容器内对进程暴露的逻辑卡通常只有一张。

当前机器的经验是：

- `ASCEND_RT_VISIBLE_DEVICES=0` 可用
- 不设也可用
- 设成物理卡号有时反而会失败

所以迁移时建议先试：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
```

## 8. 我建议的复现顺序

不要一上来就直接跑大 benchmark。更稳的顺序是：

1. 先确认 `npu-smi info` 正常
2. 下载模型和 `warmup_two_stage`
3. 先搭 `transformers + torch_npu` env
4. 确认单次 Python import 和单次模型加载通过
5. 再搭 `vllm-ascend` env
6. 先用 `python -c "import vllm; import vllm_ascend"` 做烟雾测试
7. 再启动 `vllm serve`
8. 最后再跑 benchmark

## 9. 已确认的坑

### 9.1 `transformers` 版本不能乱升

当前这套环境里：

- `transformers==4.57.1` 不兼容
- `transformers==4.55.4` 才能稳定加载当前 Curious-VLA

### 9.2 `numpy` 不能升到 `2.x`

当前这套 NPU 环境里：

- `numpy==2.x` 会破坏 `torch_npu`
- 必须回退到 `numpy==1.26.4`

### 9.3 `vllm` 不能直接按标准方式安装

要点只有一个：

- 必须用 `VLLM_TARGET_DEVICE=empty` 来安装 upstream `vllm`

### 9.4 `navsim` 统计文件路径要对

当前 agent 会读：

- `/home/ma-user/curious_vla/stats/trajectory_stats_train.json`

如果环境变量或工作目录不对，agent 初始化会失败。

## 10. 和当前仓库对应的关键文件

环境变量脚本：

- [local/local_env_npu.sh](/home/ma-user/curious_vla/local/local_env_npu.sh)
- [local/local_env_vllm_ascend.sh](/home/ma-user/curious_vla/local/local_env_vllm_ascend.sh)

主要总报告：

- [npu_adaptation_summary.md](/home/ma-user/curious_vla/latency_docs/npu_adaptation_summary.md)

本地 `transformers` benchmark：

- [run_planning_latency_benchmark_npu.py](/home/ma-user/curious_vla/local/run_planning_latency_benchmark_npu.py)
- [run_planning_latency_benchmark_npu.sh](/home/ma-user/curious_vla/local/run_planning_latency_benchmark_npu.sh)

`vllm-ascend` benchmark：

- [run_vllm_planning_latency_benchmark.py](/home/ma-user/curious_vla/local/run_vllm_planning_latency_benchmark.py)
- [run_vllm_planning_latency_benchmark.sh](/home/ma-user/curious_vla/local/run_vllm_planning_latency_benchmark.sh)
- [run_vllm_semantic_validation.py](/home/ma-user/curious_vla/local/run_vllm_semantic_validation.py)
- [run_vllm_semantic_validation.sh](/home/ma-user/curious_vla/local/run_vllm_semantic_validation.sh)

## 11. 最短复现结论

如果你只想抓住最关键的信息，可以直接记住下面这几条：

- 当前成功环境不是一套，而是两套
- 本地链路用的是 `torch 2.1.0 + torch-npu 2.1.0.post12 + transformers 4.55.4`
- 服务链路用的是 `torch 2.5.1 + torch-npu 2.5.1.post1 + transformers 4.52.4 + vllm-ascend 0.9.1 + vllm 0.9.1+empty`
- `numpy` 必须钉在 `1.26.4`
- 数据最小集合用 `maps + warmup_two_stage + test metadata`
- 在新机器上先跑通 import 和单次推理，再跑 benchmark
