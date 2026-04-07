# Curious-VLA `warmup_two_stage` Benchmark 安装与下载指南（NPU 迁移版）

本文面向后续在 **NPU 机器** 上复现 Curious-VLA benchmark 链路。  
目标不是一次性下载完整 `navtest`，而是先用更小的 **`warmup_two_stage`** 把整条链路跑通。

这份文档基于当前仓库在 CUDA 机器上的实际部署结果整理，保留了已经验证过的下载、目录和脚本组织方式，同时把 **NPU 环境必须替换的部分** 单独标出来，避免直接照搬 CUDA 安装命令。

## 1. 推荐的最小目标

推荐先跑：

- `warmup_two_stage`

不建议一开始就跑：

- `navtest`

原因：

- `navtest` 的完整 `test` 传感器数据很大，下载和整理成本高
- `warmup_two_stage` 体量小得多，更适合先验证环境、数据、服务和评测链路

## 2. 最小必需内容

要把本地 benchmark 跑通，最小需要这三类数据：

- `maps`
- `warmup_two_stage`
- `test metadata`

注意：

- 这里只需要 `test` split 的 metadata/log files
- 不需要下载完整 `test` 传感器数据

## 3. 建议目录规划

下面是本次实际使用的目录规划，后续迁移到 NPU 时建议继续沿用：

```bash
PROJECT_ROOT=/mnt/42_store/zxz/HUAWEI/VLA/curious_vla
DATA_ROOT=/mnt/42_store/zxz/HUAWEI/VLA/navsim_data

NAVSIM_ENV_PREFIX=/data/zxz/condaenv/curious_vla/navsim
LF_ENV_PREFIX=/data/zxz/condaenv/curious_vla/lf
LLAMAFACTORY_ROOT=/data/zxz/condaenv/curious_vla/src/LlamaFactory
CURIOUS_VLA_MODEL_DIR=/data/zxz/condaenv/curious_vla/models/Curious-VLA
```

如果换机器，尽量只改这些根路径，不要改仓库内脚本逻辑。

## 4. 下载项

### 4.1 下载 `maps`

```bash
mkdir -p /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads
cd /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads

wget -c https://hf-mirror.com/datasets/pengxiang/nuplan_maps/resolve/main/nuplan-maps-v1.1.zip
unzip -o nuplan-maps-v1.1.zip
rm -f nuplan-maps-v1.1.zip
mv -f nuplan-maps-v1.0 maps
```

### 4.2 下载 `warmup_two_stage`

```bash
cd /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads

wget -c https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/navsim_v2.2_warmup_two_stage.tar.gz
tar -xzvf navsim_v2.2_warmup_two_stage.tar.gz
rm -f navsim_v2.2_warmup_two_stage.tar.gz
```

解压后应包含：

```bash
downloads/warmup_two_stage/
  openscene_meta_datas/
  sensor_blobs/
  synthetic_scene_pickles/
```

### 4.3 下载 `test metadata`

```bash
cd /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads

wget -c https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz
tar -xzf openscene_metadata_test.tgz
rm -f openscene_metadata_test.tgz
mv -f openscene-v1.1/meta_datas test_navsim_logs
rm -rf openscene-v1.1
```

### 4.4 下载模型

模型仓库：

- `MashiroLn/Curious-VLA`

推荐使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
```

下载命令：

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="MashiroLn/Curious-VLA", local_dir="/data/zxz/condaenv/curious_vla/models/Curious-VLA", max_workers=1)'
```

模型下载完成后，目录中至少应看到：

- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`

## 5. 目录整理

最终建议整理成：

```bash
navsim_data/
  maps -> downloads/maps
  warmup_two_stage -> downloads/warmup_two_stage
  navsim_logs/
    test -> downloads/test_navsim_logs/test
```

手动软链接命令：

```bash
mkdir -p /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/navsim_logs

ln -sfn /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads/maps \
  /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/maps

ln -sfn /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads/warmup_two_stage \
  /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/warmup_two_stage

ln -sfn /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads/test_navsim_logs/test \
  /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/navsim_logs/test
```

这里最容易出错的是最后一个链接。  
必须指向：

```bash
downloads/test_navsim_logs/test
```

不能只指向：

```bash
downloads/test_navsim_logs
```

## 6. 环境拆分

建议保留两套环境：

- `navsim`：负责 metric caching 和 benchmark evaluation
- `lf`：负责模型服务

### 6.1 `navsim` 环境

建议：

- Python 3.9

基础安装命令：

```bash
cd /mnt/42_store/zxz/HUAWEI/VLA/curious_vla
conda env create -p /data/zxz/condaenv/curious_vla/navsim -f navsim_eval/environment.yml
conda activate /data/zxz/condaenv/curious_vla/navsim
pip install -e navsim_eval
```

在 CUDA 机器上，本次实际额外补装了 `torch`。  
迁移到 NPU 时，这里不要直接照抄 CUDA 版本，而是替换成你们 Ascend 环境对应的组合，例如：

- 匹配 CANN 的 `torch`
- `torch_npu`
- 其他 Ascend 运行时依赖

一个占位示例：

```bash
conda activate /data/zxz/condaenv/curious_vla/navsim
pip install torch==<ascend_compatible_torch>
pip install torch_npu==<ascend_compatible_torch_npu>
```

### 6.2 `lf` 环境

建议：

- Python 3.11

安装命令：

```bash
conda create -y -p /data/zxz/condaenv/curious_vla/lf python=3.11
git clone --depth 1 git@github.com:hiyouga/LlamaFactory.git /data/zxz/condaenv/curious_vla/src/LlamaFactory

conda activate /data/zxz/condaenv/curious_vla/lf
cd /data/zxz/condaenv/curious_vla/src/LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```

迁移到 NPU 时，同样需要把默认的 CUDA `torch` 栈改为 Ascend 对应版本。

## 7. 当前仓库中可复用的脚本

本次已经整理好的本地脚本如下：

- [local_env.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/local_env.sh)
- [finalize_warmup_data.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/finalize_warmup_data.sh)
- [run_metric_caching_warmup.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/run_metric_caching_warmup.sh)
- [run_warmup_eval.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/run_warmup_eval.sh)
- [wait_and_run_warmup_eval.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/wait_and_run_warmup_eval.sh)
- [start_lf_server.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/start_lf_server.sh)
- [start_lf_server_gpu1.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/start_lf_server_gpu1.sh)

迁移到 NPU 时：

- `local_env.sh` 可以继续复用
- `finalize_warmup_data.sh` 可以继续复用
- `run_metric_caching_warmup.sh` 和 `run_warmup_eval.sh` 通常也可以继续复用
- `wait_and_run_warmup_eval.sh` 可以作为自动串联脚本继续复用
- `start_lf_server_gpu1.sh` 里的设备变量需要改

## 8. 推荐执行顺序

在一台新机器上，建议按下面顺序落地：

1. 准备目录
2. 下载 `maps`
3. 下载 `warmup_two_stage`
4. 下载 `test metadata`
5. 建 `navsim` 环境
6. 建 `lf` 环境
7. 下载 Curious-VLA 模型
8. 整理软链接
9. 跑 `metric caching`
10. 启动模型服务
11. 跑 `warmup` evaluation

## 9. `metric caching` 运行方式

当前仓库可直接运行：

```bash
/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/run_metric_caching_warmup.sh
```

本次在当前机器上的实际结果是：

- `warmup_two_stage` 共缓存成功 220 个 feature/target

如果这里失败，优先检查：

- `maps` 软链接是否正确
- `warmup_two_stage` 是否完整
- `navsim_logs/test` 是否指到正确的 `.pkl` 目录

## 10. 模型服务

当前实际验证通过的策略是：

- 使用 LLaMA-Factory
- 使用 `huggingface` backend
- 不依赖 `vllm`

这是因为：

- `vllm` 在当前流程里不是默认可用项
- 先用 `huggingface` 更容易把链路跑通

CUDA 机器上的启动脚本是：

- [start_lf_server_gpu1.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/start_lf_server_gpu1.sh)

这个脚本当前写的是：

```bash
CUDA_VISIBLE_DEVICES=1
```

迁移到 NPU 时，需要替换为你们环境的设备控制变量，例如：

- `ASCEND_RT_VISIBLE_DEVICES`
- 或者实际集群要求的其他变量

同时还要确认：

- `llamafactory-cli api` 在 NPU 环境下是否能正常加载模型
- `transformers` 多模态推理是否兼容 `torch_npu`

## 11. `warmup` evaluation 运行方式

当前仓库可直接运行：

```bash
EXPERIMENT_NAME=warmup_smoke \
MODEL_NAME_OR_PATH=/data/zxz/condaenv/curious_vla/models/Curious-VLA \
/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/run_warmup_eval.sh
```

这个脚本内部会调用：

- `run_pdm_score_one_stage.py`

并使用：

- `warmup_two_stage`
- `navsim_logs/test`
- `metric_cache_warmup_two_stage`

## 12. 迁移到 NPU 时必须重新确认的点

下面这些项目在 NPU 上都不能直接假定可用，需要逐项验证：

- `torch` 与 `torch_npu` 版本组合
- Ascend 驱动与 CANN 版本
- `transformers` 是否能正常加载 Curious-VLA
- 图像输入预处理是否有算子兼容问题
- LLaMA-Factory API 服务是否能稳定启动
- OpenAI-compatible API 是否能正常返回结果

如果服务起不来，建议排查顺序是：

1. 先在 Python 里直接加载模型，确认 `transformers` + NPU 可用
2. 再确认单条图文推理可以完成
3. 最后再接入 `llamafactory-cli api`

## 13. 下载与拉取建议

### 13.1 Hugging Face

推荐统一使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 13.2 GitHub

如果 HTTP 不稳定，优先使用 SSH：

```bash
git clone git@github.com:hiyouga/LlamaFactory.git
```

## 14. 最容易踩的坑

### 14.1 `warmup_two_stage` 不是完整独立数据集

它还需要：

- `test metadata`

### 14.2 `navsim_logs/test` 软链接容易指错

正确目标是：

- `downloads/test_navsim_logs/test`

### 14.3 不要默认使用 `vllm`

先用：

- `huggingface` backend

把链路跑通之后，再考虑性能优化。

### 14.4 不要把 CUDA 命令原样搬到 NPU

尤其是下面这些都必须按 NPU 栈替换：

- `torch==2.5.1 + cu121`
- `CUDA_VISIBLE_DEVICES`
- CUDA 专用 wheel 源

## 15. 一份可执行的最小复现清单

如果你后面要在 NPU 上重新搭一套，建议直接按这个 checklist 执行：

1. 建目录：项目、数据、conda 环境、模型目录
2. 下载 `maps`
3. 下载 `warmup_two_stage`
4. 下载 `test metadata`
5. 建 `navsim` 环境并安装 `navsim_eval`
6. 建 `lf` 环境并安装 LLaMA-Factory
7. 安装 Ascend 对应的 `torch` / `torch_npu`
8. 下载 `MashiroLn/Curious-VLA`
9. 建立软链接
10. 跑 `run_metric_caching_warmup.sh`
11. 先手工验证模型能在 NPU 上完成一条推理
12. 再启动 API 服务
13. 跑 `run_warmup_eval.sh`

## 16. 相关文档

当前仓库里还有一份更偏向本机 CUDA 实操记录的文档，可作为补充参考：

- [warmup_benchmark_setup.md](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/docs/warmup_benchmark_setup.md)

如果后续你确定具体的 Ascend 驱动、CANN、`torch_npu` 版本，我可以继续把这份文档补成一份 **可直接执行的 NPU 精确安装手册**，把占位版本号都替换掉。
