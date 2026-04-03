# Curious-VLA `warmup_two_stage` Benchmark 安装与下载总结

本文记录的是在当前仓库中，**先把 NAVSIM benchmark 整个链路跑通**的最小方案。  
目标不是直接跑体量很大的 `navtest`，而是先跑通更小的 **`warmup_two_stage`**。

这份文档基于本次实际部署过程整理，重点补充了仓库原始文档里没有写清楚、但实际会踩坑的部分。

## 1. 推荐目标

如果你只是想先确认：

- 环境能否装起来
- 数据能否对齐
- metric caching 能否成功
- 模型服务能否拉起
- evaluation 脚本能否真正跑完

那么推荐先跑：

- `warmup_two_stage`

而不是：

- `navtest`

原因：

- `navtest` 需要非常大的 `test` 传感器数据，量级在 200GB 以上
- `warmup_two_stage` 本身只有约 1.2GB 级别，更适合先验证全链路

## 2. 最终链路

本次跑通 `warmup_two_stage` 的链路是：

1. 建两套 conda 环境
2. 下载 `maps`
3. 下载 `warmup_two_stage`
4. 额外下载一个较小的 `test metadata` 包
5. 整理目录结构
6. 运行 `metric caching`
7. 启动模型服务
8. 运行 `warmup` evaluation

## 3. 一个关键坑

`warmup_two_stage` **不能只下载它自己就完整跑通本地评测**。

实际本地运行时，至少还需要：

- `warmup_two_stage` 的传感器与 synthetic scenes
- `test` split 的 **metadata/log files**

但不需要：

- `test` split 的完整传感器数据

也就是说，最小方案不是：

- `warmup_two_stage` only

而是：

- `warmup_two_stage`
- `openscene_metadata_test.tgz`
- `maps`

这是本次实际验证后的结论。

## 4. 目录规划

本次使用的目录如下：

```bash
PROJECT_ROOT=/mnt/42_store/zxz/HUAWEI/VLA/curious_vla
DATA_ROOT=/mnt/42_store/zxz/HUAWEI/VLA/navsim_data

NAVSIM_ENV_PREFIX=/data/zxz/condaenv/curious_vla/navsim
LF_ENV_PREFIX=/data/zxz/condaenv/curious_vla/lf
LLAMAFACTORY_ROOT=/data/zxz/condaenv/curious_vla/src/LlamaFactory
CURIOUS_VLA_MODEL_DIR=/data/zxz/condaenv/curious_vla/models/Curious-VLA
```

如果你以后迁移到别的机器，建议只改这几个根路径。

## 5. 环境

### 5.1 `navsim` 环境

用于：

- metric caching
- benchmark evaluation

建议：

- Python 3.9

安装方式：

```bash
cd /mnt/42_store/zxz/HUAWEI/VLA/curious_vla
conda env create -p /data/zxz/condaenv/curious_vla/navsim -f navsim_eval/environment.yml
conda activate /data/zxz/condaenv/curious_vla/navsim
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e navsim_eval
```

说明：

- `navsim_eval/requirements.txt` 里没有真正安装 `torch`，需要手动补
- 当前机器驱动兼容 `cu121`，所以这里用 `torch 2.5.1 + cu121`

### 5.2 `lf` 环境

用于：

- 启动 Curious-VLA 模型服务

建议：

- Python 3.11

安装方式：

```bash
conda create -y -p /data/zxz/condaenv/curious_vla/lf python=3.11
git clone --depth 1 git@github.com:hiyouga/LlamaFactory.git /data/zxz/condaenv/curious_vla/src/LlamaFactory

conda activate /data/zxz/condaenv/curious_vla/lf
cd /data/zxz/condaenv/curious_vla/src/LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```

### 5.3 `lf` 环境的实际坑

这次实际遇到的问题：

- LLaMA-Factory 默认建议 `vllm`
- 但当前环境里没有装 `vllm`
- 更重要的是，`lf` 环境一开始装入了不兼容当前驱动的 `torch/cu13`

因此最终做法是：

- 把 `lf` 环境中的 `torch` 重新切回 `cu121`
- 服务端优先使用 `huggingface` backend，而不是 `vllm`

修正方式：

```bash
conda activate /data/zxz/condaenv/curious_vla/lf
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## 6. 下载

### 6.1 地图

官方 S3 在部分机器上可能 TLS 不稳定。  
本次实际可用方案是使用镜像：

```bash
mkdir -p /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads
cd /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads

wget -c https://hf-mirror.com/datasets/pengxiang/nuplan_maps/resolve/main/nuplan-maps-v1.1.zip
unzip -o nuplan-maps-v1.1.zip
rm -f nuplan-maps-v1.1.zip
mv -f nuplan-maps-v1.0 maps
```

### 6.2 `warmup_two_stage`

本次直接使用镜像：

```bash
cd /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads

wget -c https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/navsim_v2.2_warmup_two_stage.tar.gz
tar -xzvf navsim_v2.2_warmup_two_stage.tar.gz
rm -f navsim_v2.2_warmup_two_stage.tar.gz
```

解压后会得到：

```bash
downloads/warmup_two_stage/
  openscene_meta_datas/
  sensor_blobs/
  synthetic_scene_pickles/
```

### 6.3 `test metadata`

这是 `warmup_two_stage` 本地评测的额外依赖。  
只要 metadata，不要全量 `test` 传感器。

```bash
cd /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads

wget -c https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz
tar -xzf openscene_metadata_test.tgz
rm -f openscene_metadata_test.tgz
mv -f openscene-v1.1/meta_datas test_navsim_logs
rm -rf openscene-v1.1
```

### 6.4 模型

本次实际下载目标：

- `MashiroLn/Curious-VLA`

使用镜像：

```bash
conda activate /data/zxz/condaenv/curious_vla/lf
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="MashiroLn/Curious-VLA", local_dir="/data/zxz/condaenv/curious_vla/models/Curious-VLA", max_workers=1)'
```

说明：

- 如果镜像中途断流，可以重复执行同一条命令
- `snapshot_download` 会续传
- 本次模型最终由两块 `model-00001-of-00002.safetensors` 和 `model-00002-of-00002.safetensors` 组成

## 7. 镜像与拉取策略

### 7.1 Hugging Face

推荐统一使用：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 7.2 GitHub

如果 HTTP 拉取不稳定，优先使用 SSH：

```bash
git clone git@github.com:hiyouga/LlamaFactory.git
```

## 8. 数据目录整理

本次实际使用的整理逻辑是：

```bash
navsim_data/
  maps -> downloads/maps
  warmup_two_stage -> downloads/warmup_two_stage
  navsim_logs/
    test -> downloads/test_navsim_logs/test
```

如果手动做软链接：

```bash
mkdir -p /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/navsim_logs

ln -sfn /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads/maps \
  /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/maps

ln -sfn /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads/warmup_two_stage \
  /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/warmup_two_stage

ln -sfn /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/downloads/test_navsim_logs/test \
  /mnt/42_store/zxz/HUAWEI/VLA/navsim_data/navsim_logs/test
```

## 9. 本地环境变量

当前仓库中已经整理了一份本地配置脚本：

- [local_env.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/local_env.sh)

它的职责是固定：

- 项目路径
- 数据路径
- conda 环境路径
- 模型路径
- `warmup_two_stage` 的缓存路径和数据路径

## 10. 本次可直接复用的脚本

### 10.1 数据整理

- [finalize_warmup_data.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/finalize_warmup_data.sh)

### 10.2 缓存

- [run_metric_caching_warmup.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/run_metric_caching_warmup.sh)

说明：

- 这个脚本名字是 `warmup`
- 实际转发到 [run_metric_caching_navtest.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/run_metric_caching_navtest.sh)
- 文件名比较旧，但内容已经改成 `warmup_two_stage`

运行方式：

```bash
/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/run_metric_caching_warmup.sh
```

### 10.3 评测

- [run_warmup_eval.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/run_warmup_eval.sh)
- [wait_and_run_warmup_eval.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/wait_and_run_warmup_eval.sh)

运行方式：

```bash
EXPERIMENT_NAME=warmup_smoke \
MODEL_NAME_OR_PATH=/data/zxz/condaenv/curious_vla/models/Curious-VLA \
/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/run_warmup_eval.sh
```

如果你想让它自动等待：

- `metric caching` 结束
- 模型服务 `8192` 就绪

再触发评测，推荐改用：

```bash
/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/wait_and_run_warmup_eval.sh
```

说明：

- 这个脚本专门规避了 `pgrep` 匹配到自己导致无限等待的问题
- 默认会自动生成 `EXPERIMENT_NAME=warmup_smoke_时间戳`

### 10.4 启动服务

- 通用脚本：[start_lf_server.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/start_lf_server.sh)
- 避开 `GPU0` 的脚本：[start_lf_server_gpu1.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/start_lf_server_gpu1.sh)

当前推荐：

```bash
/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/start_lf_server_gpu1.sh
```

## 11. 为什么本次不用 `GPU0`

本次机器上：

- `GPU0` 已经被其他任务占用
- `GPU1-7` 基本空闲

所以为了避免冲突，当前服务脚本显式使用：

```bash
CUDA_VISIBLE_DEVICES=1
```

这对 3B 模型的 warmup 链路是足够的。

## 12. 评测链路的最小检查顺序

建议按这个顺序检查：

1. `maps` 是否存在
2. `warmup_two_stage` 是否存在
3. `navsim_logs/test` 是否正确指向 `.pkl` 文件目录
4. `metric caching` 是否完成
5. 模型两块 `safetensors` 是否完整
6. 模型服务的 `8192` 端口是否能返回 `/v1/models`
7. 再跑 `warmup` eval

## 13. 本次实际踩过的坑

### 13.1 官方地图源 TLS 失败

现象：

- 官方 `motional` S3 链接 TLS 握手失败

解决：

- 换 `hf-mirror` 上的镜像地图包

### 13.2 `warmup_two_stage` 并不能单独完成本地评测

现象：

- `metric caching` 一开始出现 `0 files`

原因：

- 缺少 `test metadata`

解决：

- 单独补 `openscene_metadata_test.tgz`

### 13.3 `test metadata` 解压后目录多一层

现象：

- `navsim_logs/test` 指过去后只有一个子目录 `test`
- loader 读不到 `.pkl`

解决：

- 软链接要指向：

```bash
downloads/test_navsim_logs/test
```

而不是：

```bash
downloads/test_navsim_logs
```

### 13.4 `lf` 环境的 CUDA/torch 版本不兼容

现象：

- `torch` 初始化报驱动过旧

原因：

- 环境里混入了 `cu13` 版本

解决：

- 重装成 `torch 2.5.1 + cu121`

### 13.5 `vllm` 不是默认可用

现象：

- `llamafactory-cli api` 报缺少 `vllm`

解决：

- 优先改成 `huggingface` backend
- 先把链路跑通，再考虑是否切回 `vllm`

## 14. 迁移到 NPU 时要改什么

这份文档记录的是 **当前这台 CUDA 机器上已经验证过的方案**。  
迁移到 NPU 时，不建议照搬 CUDA 相关安装项。

你需要重点改这几部分：

### 14.1 `torch` 安装

当前文档里用的是：

```bash
torch==2.5.1 + cu121
```

迁移到 NPU 时，通常需要替换为：

- Ascend 对应版本的 `torch`
- `torch_npu`
- 匹配的 CANN / 驱动

### 14.2 设备环境变量

当前脚本里使用：

```bash
CUDA_VISIBLE_DEVICES=1
```

迁移到 NPU 时要改成 NPU 对应的设备变量，例如：

- `ASCEND_RT_VISIBLE_DEVICES`
- 或你们环境要求的其他变量

### 14.3 模型服务后端

当前推荐的是：

- `huggingface` backend

在 NPU 上：

- 不要默认假设 `vllm` 可用
- 先验证 `transformers + torch_npu` 能跑通
- 再考虑是否换更高性能推理后端

### 14.4 需要重新验证的部分

以下内容在 NPU 上需要重新验证，不应直接假定可用：

- 多模态模型加载
- 图像输入预处理
- LLaMA-Factory API 服务
- OpenAI-compatible API 是否正常返回

## 15. 推荐的最小执行顺序

在新机器上，建议按以下顺序执行：

1. 建 `navsim` 环境
2. 建 `lf` 环境
3. 下载 `maps`
4. 下载 `warmup_two_stage`
5. 下载 `test metadata`
6. 运行 `finalize_warmup_data.sh`
7. 运行 `run_metric_caching_warmup.sh`
8. 下载模型
9. 启动服务
10. 运行 `run_warmup_eval.sh`

## 16. 当前文档适用范围

这份文档适用于：

- Linux
- conda
- Curious-VLA 当前仓库
- 先跑通 `warmup_two_stage`

它不直接保证以下场景一定可用：

- NPU 环境零修改可跑
- `vllm` 零修改可跑
- `navtest` 全量数据路线

如果后续你真的切到 NPU，建议基于本文再新增一份：

- `docs/warmup_benchmark_setup_npu.md`

专门记录：

- `torch_npu`
- CANN
- NPU 设备变量
- NPU 推理后端
