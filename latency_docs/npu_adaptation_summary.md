# Curious-VLA NPU 适配总文档

本文汇总当前仓库为 **Ascend NPU** 运行 Curious-VLA latency 相关测评所做的主要适配，覆盖两条已经实际打通的路径：

- 本地进程内 `transformers + torch_npu`
- 服务化 `vllm-ascend`

本文是当前 NPU 侧的主报告，合并了原先独立的：

- `vllm_ascend_validation.md`
- `vllm_ascend_1280x704_total_report.md`

如果只看一份文档，优先看本文。

## 1. 目标范围

本轮工作的目标是：

- 在没有 CUDA GPU、只有 Ascend NPU 的机器上跑通 Curious-VLA latency 测试
- 先完成模型侧和 agent 侧的基础 benchmark
- 再补一层“结果合理性验证”，避免只测到无意义输出
- 尽量补到更接近真实 planning 路径的统计

当前已经实际跑通的内容包括：

- model-only latency benchmark
- `AgentInput -> NavsimCoTQwenAgent.compute_trajectory()` planning latency benchmark
- `vllm-ascend` OpenAI-compatible 服务
- 基于真实 warmup scene 的轻量级语义 gate
- `1280x704` 分辨率下的 `vllm-ascend` planning latency benchmark

当前还没有补全的内容包括：

- 完整 PDM / EPDMS 质量评测
- 全量 `navtest` 数据集评测
- `SceneLoader + sequential worker + 全套 PDM eval` 的统一时延统计
- 面向长期部署稳定性的系统化压测

## 2. 机器与运行时基线

当前机器环境：

- OS：`EulerOS 2.0 (SP10)`
- 架构：`aarch64`
- NPU：`Ascend 910B3`
- `npu-smi`：`23.0.6`
- CANN：`8.1.RC1`

NPU 可用性已通过 `npu-smi info` 确认。

## 3. 存储与目录规划

由于 `/home/ma-user/work` 剩余空间不足以容纳模型和数据，本次所有大文件统一放在：

```bash
/cache/ma-user/curious_vla_assets
```

当前实际使用的目录：

```bash
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

兼容原仓库访问方式时，还使用了这些入口：

- `/cache/ma-user/curious_vla_assets/data/maps`
- `/cache/ma-user/curious_vla_assets/data/warmup_two_stage`
- `/cache/ma-user/curious_vla_assets/data/navsim_logs/test`

## 4. 已下载资源

当前已就位资源如下。

模型：

- `/cache/ma-user/curious_vla_assets/models/Curious-VLA`

数据：

- warmup 数据：
  `/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage`
- 地图：
  `/cache/ma-user/curious_vla_assets/data/downloads/maps`
- test metadata：
  `/cache/ma-user/curious_vla_assets/data/downloads/test_navsim_logs`

推荐继续使用 HF 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 4.1 本次使用的模型细节

本次 benchmark 实际使用的模型目录是：

- `/cache/ma-user/curious_vla_assets/models/Curious-VLA`

模型卡和仓库信息表明：

- 模型名：`MashiroLn/Curious-VLA`
- 模型架构：`Qwen2_5_VLForConditionalGeneration`
- 项目 README 标注的基座：`Qwen2.5-VL-3B-Instruct`

从本地 `config.json` 可以确认的关键结构包括：

- `hidden_size=2048`
- `intermediate_size=11008`
- `num_hidden_layers=36`
- `num_attention_heads=16`
- `num_key_value_heads=2`

因此当前可以把它理解为：

- 一个基于 `Qwen2.5-VL-3B` 路线微调得到的 Curious-VLA 模型

本地模型目录的磁盘占用约为：

- `7.6G`

主要权重文件为：

- `model-00001-of-00002.safetensors`：`4.7G`
- `model-00002-of-00002.safetensors`：`3.0G`

这里的 `7.6G` 是磁盘占用，不等于严格意义上的参数量；参数规模仍以 `3B` 级别理解更合适。

### 4.2 当前推荐的最小数据目标

如果目标是先把 Curious-VLA 的 benchmark 链路在 NPU 上跑通，当前最推荐的最小目标不是完整 `navtest`，而是：

- `warmup_two_stage`

原因：

- `warmup_two_stage` 体量小，适合先验证环境、模型、数据布局和 benchmark 链路
- 完整 `navtest` 的传感器数据更大，下载和整理成本明显更高
- 当前这轮 NPU latency 适配本身也主要围绕 `warmup_two_stage` 展开

当前最小必需数据是三类：

- `maps`
- `warmup_two_stage`
- `test metadata`

注意：

- 这里只需要 `test` split 的 metadata / log files
- 不需要下载完整 `test` 传感器数据
- `warmup_two_stage` 不是完整独立数据集，它仍然依赖 `test metadata`

### 4.3 推荐下载方式

推荐统一使用 Hugging Face 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
```

当前这批资源对应的下载来源可以概括为：

- `maps`
  - `pengxiang/nuplan_maps`
- `warmup_two_stage`
  - `OpenDriveLab/OpenScene`
- `test metadata`
  - `OpenDriveLab/OpenScene`
- 模型
  - `MashiroLn/Curious-VLA`

如果需要在另一台机器上重新拉一套当前使用的数据，下面这些命令是最接近当前方案的参考版本：

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

模型下载可参考：

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="MashiroLn/Curious-VLA", local_dir="/cache/ma-user/curious_vla_assets/models/Curious-VLA", max_workers=1)'
```

### 4.4 数据布局与软链接注意事项

当前主文档前面提到的这三个入口：

- `/cache/ma-user/curious_vla_assets/data/maps`
- `/cache/ma-user/curious_vla_assets/data/warmup_two_stage`
- `/cache/ma-user/curious_vla_assets/data/navsim_logs/test`

本质上是为了兼容仓库原有的数据访问方式。

推荐整理成：

```bash
/cache/ma-user/curious_vla_assets/data/
  maps -> downloads/maps
  warmup_two_stage -> downloads/warmup_two_stage
  navsim_logs/
    test -> downloads/test_navsim_logs/test
```

如果需要手工重建，可参考：

```bash
mkdir -p /cache/ma-user/curious_vla_assets/data/navsim_logs

ln -sfn /cache/ma-user/curious_vla_assets/data/downloads/maps \
  /cache/ma-user/curious_vla_assets/data/maps

ln -sfn /cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage \
  /cache/ma-user/curious_vla_assets/data/warmup_two_stage

ln -sfn /cache/ma-user/curious_vla_assets/data/downloads/test_navsim_logs/test \
  /cache/ma-user/curious_vla_assets/data/navsim_logs/test
```

这里最容易踩的坑是最后一个软链接。

必须指向：

- `downloads/test_navsim_logs/test`

不能只指向：

- `downloads/test_navsim_logs`

### 4.5 NAVSIM 各种规模数据集的详细解释

这一节专门解释当前项目里最容易让人混淆的几类数据名称：`OpenScene`、`mini`、`trainval`、`test`、`navtrain`、`navtest`、`navhard_two_stage`、`warmup_two_stage`、`private_test_hard_two_stage`。

先说最核心的一层关系：

- 最底层的数据来源是 `OpenScene`
- `OpenScene` 本身又分成标准数据切分：
  - `mini`
  - `trainval`
  - `test`
- NAVSIM 在这些标准切分之上，又定义了一批“更适合训练 / 评测 / 比赛”的过滤或挑战切分：
  - `navtrain`
  - `navtest`
  - `navhard_two_stage`
  - `warmup_two_stage`
  - `private_test_hard_two_stage`

再进一步说：

- `OpenScene` 更像“原始可下载标准数据”
- `NAVSIM split` 更像“基于这些原始数据组织出来的标准训练 / 测试 / 挑战集合”

#### 4.5.1 `OpenScene` 是什么

`OpenScene` 可以理解为：

- 面向 NAVSIM / 自动驾驶规划任务整理过的一套数据发布形式
- 它对应 nuPlan，但做了降采样与重组织
- 当前官方说明里强调它是从 nuPlan 压缩整理过来的，并以 `2Hz` 频率提供

对当前项目最重要的理解是：

- 你在仓库里看到的大多数 `openscene_meta_datas/*.pkl`
- 以及对应的 `sensor_blobs`

本质上都属于这条数据体系。

#### 4.5.2 `mini`、`trainval`、`test` 是什么

这三个是 `OpenScene` 的标准切分。

`mini`：

- 演示 / demo 用的小切分
- 适合验证代码能不能跑
- 不适合作为严肃 benchmark 基线

`trainval`：

- 大规模训练与验证切分
- 适合训练 agent 或做更系统的离线实验
- 但传感器数据体量非常大

`test`：

- 标准测试切分
- 比 `trainval` 小很多，但完整传感器数据依然很大
- 很多 NAVSIM 测试切分都建立在它之上

如果只想记一句：

- `mini / trainval / test` 是原始标准数据切分

#### 4.5.3 `navtrain` 是什么

`navtrain` 是 NAVSIM 定义的标准训练切分。

它的特点是：

- 基于 `trainval`
- 不是简单等于整个 `trainval`
- 而是从中筛出更有价值、更多“非平凡驾驶场景”的子集

这样做的意义是：

- 训练更聚焦
- 存储成本比完整 `trainval` 低
- 更适合作为 NAVSIM 体系里的标准训练入口

所以：

- 如果你是做训练，`navtrain` 通常比直接全量啃 `trainval` 更有针对性

#### 4.5.4 `navtest` 是什么

`navtest` 是 NAVSIM v1 体系里的标准测试切分。

它的特点是：

- 基于 `test`
- 通过 scene filter 从 `test` 中选出标准化测试场景
- 更偏“官方定义的标准测试集”

对本项目来说要特别注意：

- `navtest` 不是当前这轮 NPU latency 适配的主线数据
- 它更适合完整评测、PDM 评分、标准测试
- 如果一上来就走 `navtest` 全量传感器路线，数据体量会明显变大

所以当前主文档一直强调：

- 先用 `warmup_two_stage` 跑通链路
- 不要一开始就冲完整 `navtest`

#### 4.5.5 `navhard_two_stage` 是什么

`navhard_two_stage` 是 NAVSIM v2 更偏正式本地评测的一条测试切分。

可以把它理解成：

- 基于 `test` 的 harder split
- 面向 NAVSIM v2 的 pseudo closed-loop / two-stage 评测
- 包含真实与合成场景成分

它比 `warmup_two_stage` 更接近正式挑战评测环境，但代价是：

- 数据更大
- 链路更复杂
- 对环境和脚本稳定性的要求更高

如果后面你要把当前 NPU 侧 benchmark 往“更正式的本地评测”推进，`navhard_two_stage` 会比 `warmup_two_stage` 更值得关注。

#### 4.5.6 `warmup_two_stage` 是什么

`warmup_two_stage` 是当前最重要的一类数据，因为这轮 NPU benchmark 主要就是围绕它跑通的。

它可以理解成：

- Hugging Face warmup leaderboard 对应的 warmup 测试切分
- 一个更小、更轻量、但仍然足以验证两阶段规划链路的数据集

它的优点是：

- 体量小
- 下载快
- 本地环境更容易先跑通
- 官方语义上也很明确：
  本地在 `warmup_two_stage` 跑出的结果，应该和 warmup leaderboard 上看到的结果更接近

但它也有边界：

- 它不是完整大规模测试集
- 当前我们手上的 warmup scene 每个样本 future 标注很短
- 对 latency 很适合，对完整规划质量门槛不够

对当前项目最实际的结论就是：

- 如果你的目标是先学会项目、先把环境跑通、先得到一组可工作的 latency 数字，优先用 `warmup_two_stage`

#### 4.5.7 `private_test_hard_two_stage` 是什么

`private_test_hard_two_stage` 是比赛 / challenge 里的私有测试切分。

它的定位是：

- 真正用于官方挑战榜单的私有评测数据
- 本地不能像普通公开数据那样随便直接完整验证最终成绩

所以它对当前本地 NPU 适配的意义主要是：

- 让你知道项目后续正式 submission 面向的是哪类数据
- 但它不是当前最适合做本地 latency 适配起点的选择

#### 4.5.8 为什么会觉得“好多数据名字很乱”

因为这里实际上混了两层命名：

- 一层是原始数据发布切分：
  - `mini`
  - `trainval`
  - `test`
- 一层是 NAVSIM / 挑战定义的训练、测试、比赛切分：
  - `navtrain`
  - `navtest`
  - `navhard_two_stage`
  - `warmup_two_stage`
  - `private_test_hard_two_stage`

再加上当前仓库同时涉及：

- 训练
- 本地评测
- challenge 提交
- latency benchmark

所以看起来会像是“同一个项目里有好几套数据名词”。

更简单的记法是：

- 想训练：先看 `navtrain`
- 想做标准公开测试：看 `navtest`
- 想做 NAVSIM v2 更正式的 two-stage 本地测试：看 `navhard_two_stage`
- 想先轻量跑通链路 / 对 warmup 榜单口径：看 `warmup_two_stage`
- 想理解原始公开数据来源：看 `OpenScene` 的 `mini / trainval / test`

#### 4.5.9 结合当前 Curious-VLA 项目的推荐理解

如果站在“快速掌握 Curious-VLA 项目”的角度，我建议这样理解这几类数据：

1. `OpenScene`
   是整个项目数据世界的底座。
2. `navtrain`
   是更像训练入口的数据。
3. `navtest` / `navhard_two_stage`
   是更像正式评测入口的数据。
4. `warmup_two_stage`
   是当前最适合做环境验证、NPU 迁移、benchmark 试跑的数据。

所以当前这份主文档和本轮 NPU 适配，重点反复围绕 `warmup_two_stage`，不是因为它最完整，而是因为：

- 它是当前最实用、最容易落地的起点。

## 5. 可工作的 NPU 环境

### 5.1 本地 `transformers` benchmark 环境

当前实际可工作的 conda prefix：

- `/cache/ma-user/curious_vla_assets/envs/curious-vla-npu-latency`

关键兼容版本：

- `torch==2.1.0`
- `torch_npu==2.1.0.post12`
- `transformers==4.55.4`
- `numpy==1.26.4`

其他关键包：

- `accelerate==1.0.1`
- `openai==2.30.0`
- `sentencepiece==0.2.0`
- `hydra-core==1.2.0`
- `pyquaternion==0.9.9`
- `pytorch-lightning==2.2.1`
- `shapely==2.0.6`
- `geopandas==0.14.4`
- `pyproj==3.7.1`
- `pyogrio==0.12.1`
- `rasterio==1.4.4`
- `aioboto3==15.5.0`

已确认的坑：

- `transformers==4.57.1` 与当前 `torch/torch_npu` 组合不兼容
- `transformers==4.55.4` 可以正常加载 `Qwen2_5_VLForConditionalGeneration`
- GIS 相关安装过程可能把 `numpy` 升到 `2.x`
- `numpy==2.x` 会破坏当前 `torch_npu`
- 回退到 `numpy==1.26.4` 后环境恢复可用

### 5.2 `vllm-ascend` 服务环境

当前实际可工作的 conda prefix：

- `/cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend`

关键版本：

- `vllm==0.9.1+empty`
- `vllm-ascend==0.9.1`
- `torch==2.5.1`
- `torch_npu==2.5.1.post1`
- `transformers==4.52.4`
- `numpy==1.26.4`

这里单独使用一个 env，避免污染本地 `transformers` latency 环境。

### 5.3 迁移到新 NPU 机器时优先确认的点

如果后面要在另一台 Ascend 机器上重新搭一套，不要默认当前经验可以无条件复用。最先需要逐项确认的是：

- `torch` 与 `torch_npu` 的版本组合
- Ascend 驱动与 CANN 版本
- `transformers` 是否能正常加载 Curious-VLA
- 图像预处理与多模态输入是否存在算子兼容问题
- `vllm-ascend` 或其他服务后端是否能稳定返回结果

建议排查顺序：

1. 先在 Python 中直接加载模型，确认 `transformers + NPU` 可用。
2. 再确认单条图文推理可以完成。
3. 然后再接入 agent 层 benchmark。
4. 最后再考虑服务化部署或更完整的 evaluation 链路。

## 6. `navsim` / `nuplan` 导入链适配

为了让 planning latency 跑到 agent 层，除了模型本身，还补齐了 `navsim` 和 `nuplan` 导入链所需依赖。

当前已确认下面这些导入可用：

- `aioboto3`
- `nuplan`
- `navsim.common.dataclasses`
- `navsim.common.dataloader`
- `navsim.agents.curious_vla.navsim_qwen_norm_agent_cot`

其中一个关键问题不是 GIS，而是默认统计文件路径：

- `navsim_qwen_norm_agent_cot.py` 会读取 `../stats/trajectory_stats_train.json`
- 如果当前工作目录不对，会找不到该文件

因此本次通过环境变量固定：

```bash
export STATS_PATH=/home/ma-user/curious_vla/stats/trajectory_stats_train.json
```

## 7. 新增和使用的本地脚本

### 7.1 本地 NPU 环境入口

新增：

- [local/local_env_npu.sh](/home/ma-user/curious_vla/local/local_env_npu.sh)

作用：

- 统一 `PROJECT_ROOT`
- 统一 asset root
- 指定 conda env prefix
- 指定模型目录
- 指定 warmup 数据目录
- 指定 `STATS_PATH`
- 指定 `OPENSCENE_DATA_ROOT`
- 指定 `WARMUP_SENSOR_PATH`

### 7.2 `transformers` model-only latency benchmark

新增：

- [local/run_latency_benchmark_npu.py](/home/ma-user/curious_vla/local/run_latency_benchmark_npu.py)
- [local/run_latency_benchmark_npu.sh](/home/ma-user/curious_vla/local/run_latency_benchmark_npu.sh)

作用：

- 直接加载本地 Curious-VLA 模型
- 读取一张 warmup 前视图图片
- 在 NPU 上运行 `Qwen2_5_VLForConditionalGeneration.generate()`
- 输出 `mean/min/max/p50/p95`

脚本内还补了一个兼容 patch：

- 给 `torch 2.1.0` 补 `torch.compiler.is_compiling`

原因是当前 `transformers` 路径会访问这个接口，而 `torch 2.1.0` 上不完整。

### 7.3 `transformers` planning latency benchmark

新增：

- [local/run_planning_latency_benchmark_npu.py](/home/ma-user/curious_vla/local/run_planning_latency_benchmark_npu.py)
- [local/run_planning_latency_benchmark_npu.sh](/home/ma-user/curious_vla/local/run_planning_latency_benchmark_npu.sh)

这套脚本测的是更接近 end-to-end planning latency 的本地路径：

1. 从 `openscene_meta_datas/*.pkl` 读取真实 warmup scene
2. 用 `AgentInput.from_scene_dict_list(...)` 构建真实 history 输入
3. 构造最小 `SceneMetadata` stub
4. 调 `NavsimCoTQwenAgent.compute_trajectory()`
5. 用本地 `transformers + torch_npu` backend 替代远程 OpenAI-compatible server
6. 记录：
   - `compute_trajectory` 总耗时
   - backend 输入准备耗时
   - NPU generate 耗时
   - decode / parse 耗时
   - 近似 `agent_overhead_sec`

### 7.4 `vllm-ascend` 环境与 benchmark 脚本

新增：

- [local/local_env_vllm_ascend.sh](/home/ma-user/curious_vla/local/local_env_vllm_ascend.sh)
- [local/run_vllm_semantic_validation.py](/home/ma-user/curious_vla/local/run_vllm_semantic_validation.py)
- [local/run_vllm_semantic_validation.sh](/home/ma-user/curious_vla/local/run_vllm_semantic_validation.sh)
- [local/run_vllm_planning_latency_benchmark.py](/home/ma-user/curious_vla/local/run_vllm_planning_latency_benchmark.py)
- [local/run_vllm_planning_latency_benchmark.sh](/home/ma-user/curious_vla/local/run_vllm_planning_latency_benchmark.sh)
- [local/run_planning_latency_benchmark.sh](/home/ma-user/curious_vla/local/run_planning_latency_benchmark.sh)

其中统一入口是：

- [local/run_planning_latency_benchmark.sh](/home/ma-user/curious_vla/local/run_planning_latency_benchmark.sh)

它支持：

- `--backend transformers`
- `--backend vllm`

因此现在切换后端不需要再手动换脚本名。

### 7.5 更完整 warmup eval 链路的关系

除了当前已经在用的 latency benchmark 脚本，仓库里还保留了一批更偏完整评测链路的历史脚本，例如：

- [local/finalize_warmup_data.sh](/home/ma-user/curious_vla/local/finalize_warmup_data.sh)
- [local/run_metric_caching_warmup.sh](/home/ma-user/curious_vla/local/run_metric_caching_warmup.sh)
- [local/run_warmup_eval.sh](/home/ma-user/curious_vla/local/run_warmup_eval.sh)
- [local/wait_and_run_warmup_eval.sh](/home/ma-user/curious_vla/local/wait_and_run_warmup_eval.sh)

这些脚本代表的是更完整的 warmup evaluation 思路：

- 先准备数据与软链接
- 再做 metric caching
- 然后启动模型服务
- 最后跑 warmup evaluation

但要注意：

- 当前这轮 NPU 适配和 latency 结论主要不是靠这条完整链路得出的
- 其中部分脚本仍带有历史绝对路径或旧运行假设，不能保证在当前机器上直接可复用
- 所以当前主线仍然是：
  - 本地 `transformers + torch_npu` latency benchmark
  - `vllm-ascend` 语义 gate + latency benchmark

如果以后要继续补完整 warmup / PDM eval，推荐把这些脚本当作参考入口，而不是直接把它们当成当前已验证的 NPU 精确方案。

## 8. benchmark 输入输出的实际含义

这一节只解释当前仓库里已经在跑的两条 benchmark 的输入和输出语义。

### 8.1 model-only benchmark 的输入

对应脚本：

- [local/run_latency_benchmark_npu.py](/home/ma-user/curious_vla/local/run_latency_benchmark_npu.py)

这条 benchmark 的输入不是完整场景，而是：

- 一张前视图图片
- 一段固定手写 prompt

脚本里 prompt 的语义是：

- 当前只有 1 张 front-view image
- high-level intent 被写死成 `go straight`
- 历史轨迹不是从真实 scene 动态构造，而是固定模板

所以这条 benchmark 真正在测的是：

- “单张图 + 固定 prompt” 下，Curious-VLA 模型本体在 NPU 上生成一次要多久

它不是完整 planning 输入。

### 8.2 model-only benchmark 的输出

输出主要是：

- 模型生成文本
- latency 统计

报告里常见字段包括：

- `latencies_sec`
- `mean_sec`
- `p50_sec`
- `p95_sec`
- `output_preview`

这里的 `output_preview` 只是文本预览，不代表轨迹一定被 agent 成功消费。

### 8.3 planning benchmark 的输入

对应脚本：

- [local/run_planning_latency_benchmark_npu.py](/home/ma-user/curious_vla/local/run_planning_latency_benchmark_npu.py)

这条 benchmark 的输入来自真实 warmup scene：

- 一个 `openscene_meta_datas/*.pkl`
- 当前每个样本总共 `5` 帧
- 前 `4` 帧用于构建 history / current input
- 第 `5` 帧不送进模型，只作为当前可用真值用于校验

真正喂给模型的信息包括：

- 当前时刻前视图 `cam_f0`
- 过去 4 帧推出来的历史 ego pose
- 当前 high-level driving command
- agent 自动拼出的 CoT prompt

因此它的语义是：

- 测“真实场景片段输入到 planner agent 后，返回未来轨迹要多久”

### 8.4 planning benchmark 的输出

模型原始输出是一个长 JSON，包含：

- `critical_objects`
- `explanation`
- `meta_behaviour`
- `future_trajectory`

其中真正用于规划的是 `future_trajectory`。

它的语义是：

- 未来 `4s`
- `2Hz`
- 共 `8` 个点
- 每个点是 `(x, y, heading)`
- 坐标系是相对当前 ego 的局部坐标

注意：

- 模型输出的是归一化轨迹
- agent 会再用 [stats/trajectory_stats_train.json](/home/ma-user/curious_vla/stats/trajectory_stats_train.json) 做反归一化
- 最终返回的是 `Trajectory`

因此报告中的：

- `trajectory_preview`

表示已经反归一化后的最终预测轨迹前几个点。

此外，本地 `transformers` planning benchmark 里的：

- `agent_overhead_sec`

不是模型 forward 本身，而更接近：

```text
compute_trajectory 总耗时 - 本地后端 forward 总耗时
```

它主要反映的是：

- prompt 组织
- 输入准备
- agent 层封装
- 轨迹解析与后处理

### 8.5 当前 validation 字段的含义

planning benchmark 现在额外加入了结果合理性检查。

每个样本的 `validation` 字段含义如下：

- `non_fallback`
  这次是否没有退回 constant-velocity fallback
- `shape_ok`
  解析出的轨迹形状是否为 `(8, 3)`
- `finite_ok`
  轨迹里是否没有 `nan/inf`
- `sanity_valid`
  轨迹在数值上是否不过分离谱
- `reference_valid`
  是否通过了当前可用真值的误差检查
- `validated`
  是否同时满足上面这些有效性条件

其中 `sanity_valid` 当前检查：

- 单步位移是否过大
- 最终位移是否过大
- 航向角绝对值是否异常
- 相邻点 heading 跳变是否异常

`reference_valid` 当前依赖 warmup scene 里唯一可用的 future 帧，因此目前最稳定的是：

- `0.5s` 第一个 future 点误差

相关字段包括：

- `first_step_error_m`
- `first_step_yaw_error_rad`
- `ade_m`
- `fde_m`

由于当前 warmup 数据每个样本只有 `1` 个 future 点，所以目前：

- `ade_m`
- `fde_m`

在大多数情况下会等于 first-step 误差。

### 8.6 如何理解本地 planning benchmark

这条 benchmark 最适合回答的问题是：

- 在当前 NPU 机器上，使用本地 `transformers + torch_npu` 路径时，更接近真实 planning 的耗时大概是多少？

它不适合直接回答：

- 服务端 API latency 是多少？
- `vllm` 路径是否更快？
- 完整评测链路的总耗时是多少？

如果目标变成：

- 比较 `transformers` 和 `vllm` 两种后端
- 以服务化请求 latency 为主要指标

那么更应该看本文后面的 `vllm-ascend` 结果与两条路径的对比部分，而不是只看本地 planning benchmark。

## 9. `transformers + torch_npu` 路径结果

### 9.1 model-only latency

已成功运行：

```bash
./local/run_latency_benchmark_npu.sh \
  --warmup-runs 1 \
  --benchmark-runs 2 \
  --max-new-tokens 64
```

结果：

- resize：`1280x704`
- 平均 latency 约 `42.7s`

报告：

- `/cache/ma-user/curious_vla_assets/logs/latency_benchmark_npu_20260406_000347.json`

### 9.2 planning latency，固定 `1280x704` 的早期非 fallback 结果

已成功运行：

```bash
./local/run_planning_latency_benchmark_npu.sh \
  --scene-path /cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/openscene_meta_datas/086487f683be38e1b.pkl \
  --warmup-runs 0 \
  --benchmark-runs 1 \
  --max-new-tokens 512
```

结果：

- fixed resize `1280x704`
- 非 fallback
- 单场景 planning latency 约 `346.97s`

报告：

- `/cache/ma-user/curious_vla_assets/logs/planning_latency_benchmark_npu_20260406_100458.json`

### 9.3 planning latency，原始 `1920x1080`，非 fallback

已成功运行：

```bash
./local/run_planning_latency_benchmark_npu.sh \
  --scene-path /cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/openscene_meta_datas/086487f683be38e1b.pkl \
  --warmup-runs 0 \
  --benchmark-runs 1 \
  --max-new-tokens 512 \
  --use-raw-resolution
```

结果：

- raw `1920x1080`
- 非 fallback
- 单场景 planning latency 约 `611.01s`

报告：

- `/cache/ma-user/curious_vla_assets/logs/planning_latency_benchmark_npu_20260406_101232.json`

### 9.4 planning latency，固定 `1280x704`，加入 validation 后的当前基线

当前更有代表性的命令：

```bash
./local/run_planning_latency_benchmark_npu.sh \
  --scene-path /cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/openscene_meta_datas/086487f683be38e1b.pkl \
  --warmup-runs 0 \
  --benchmark-runs 1
```

运行日期：

- 2026 年 4 月 7 日

结果：

- fixed resize `1280x704`
- `fallback=False`
- `validated=True`
- total latency 约 `296.14s`
- client latency 约 `296.11s`
- `first_step_error_m ≈ 0.0134`
- `first_step_yaw_error_rad ≈ 0.00539`

报告：

- `/cache/ma-user/curious_vla_assets/logs/planning_latency_benchmark_npu_20260407_090123.json`

这条结果比早期“只看是否跑完”更有意义，因为它满足了当前定义的基本合理性门槛。

### 9.5 token 长度与 fallback 行为

planning benchmark 中一个关键现象是：

- `64` new tokens 不够
- `256` new tokens 仍然不够
- `512` new tokens 才在实测 scene 上避免 fallback

原因：

- 当前 CoT prompt 输出很长
- 模型会先输出 `critical_objects / explanation / meta_behaviour`
- `future_trajectory` 经常在字符串中途被截断
- `NavsimCoTQwenAgent` 解析失败后会退回 constant-velocity fallback

因此，如果要测真实轨迹输出，不建议把 `--max-new-tokens` 设为 `512` 以下。

## 10. `vllm-ascend` 路径适配与结果

### 10.1 为什么标准安装不行

直接执行：

```bash
pip install vllm==0.9.1
```

在当前机器上不能直接成功，主要原因有两个。

第一，没有可直接使用的 `aarch64` wheel：

- PyPI 没有当前机器可直接用的 upstream `vllm 0.9.1` wheel
- 只能拿到源码包或 `x86_64` wheel

第二，默认源码编译会走到 upstream CPU custom ops：

- 当前系统 `g++` 为 `7.3.0`
- 编译阶段会报：
  `invalid feature modifier in '-march=armv8.2-a+dotprod+fp16'`

### 10.2 当前可用的安装策略

最终可用方案是：

1. 先安装 `vllm-ascend==0.9.1`
2. 再安装 upstream `vllm`，但将 `VLLM_TARGET_DEVICE` 设为 `empty`

核心思路：

- `VLLM_TARGET_DEVICE=empty` 会跳过 upstream custom-op 编译
- `vllm-ascend` 仍然提供 Ascend 所需的插件和补丁
- 这样就绕开了 `aarch64` 上的 CPU custom-op 编译问题

实际安装流程：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate /cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend

pip install vllm-ascend==0.9.1
pip download --no-deps vllm==0.9.1
tar -xf vllm-0.9.1.tar.gz
export VLLM_TARGET_DEVICE=empty
pip install --no-deps ./vllm-0.9.1
```

之后还补装了服务所需依赖，例如：

- `fastapi[standard]`
- `openai`
- `aiohttp`
- `tiktoken`
- `msgspec`
- `sentencepiece`
- `protobuf`
- `opencv-python-headless`

### 10.3 设备映射的关键细节

这台机器虽然在 `npu-smi` 中显示物理卡为 `NPU 2`，但当前容器对进程暴露的逻辑卡号只有：

- `0`

因此当前实际结论是：

- `ASCEND_RT_VISIBLE_DEVICES=0` 可用
- 不设置也可用
- `ASCEND_RT_VISIBLE_DEVICES=2` 不可用

### 10.4 烟雾测试与服务打通

当前 env 下，这些动作已经成功：

```bash
python -c "import vllm; import vllm_ascend"
vllm --help
```

此外，下面这类命令可以成功拉起基础服务：

```bash
vllm serve /cache/ma-user/curious_vla_assets/models/Curious-VLA \
  --host 127.0.0.1 \
  --port 18000 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

并且：

- `/v1/models` 可以返回
- `/v1/chat/completions` 可以返回结果

这说明系统层面的服务路径已经打通：

- 进程能起
- 模型能加载
- scheduler 能工作
- token 生成链路能工作

### 10.5 为什么还要补“语义 gate”

服务能返回，并不代表结果适合拿来测 planning latency。

主要原因：

- Curious-VLA 是视觉语言规划模型，不是普通文本聊天模型
- 纯文本 smoke test 不能代表真实规划 prompt
- 如果结果格式不稳定或轨迹无意义，测出来的 latency 没有参考价值

因此后来补了一套轻量级语义 gate，入口是：

- [local/run_vllm_semantic_validation.py](/home/ma-user/curious_vla/local/run_vllm_semantic_validation.py)
- [local/run_vllm_semantic_validation.sh](/home/ma-user/curious_vla/local/run_vllm_semantic_validation.sh)

这套 gate 分三层：

1. text-only schema control
2. text-only planning-style control
3. 基于真实 warmup scene 和真实前视图图像的 VL planning 校验

主要检查项：

- 严格 JSON 合同
- `critical_objects` 是否完整
- `meta_behaviour` 是否在合法值域
- scene intent 和输出 `command` 是否对齐
- 轨迹能否被解析
- 反归一化轨迹是否合理
- 和 warmup 唯一 future 点的一步误差是否在阈值内

### 10.6 为什么 `1280x704` 比 `960x540` 更难

对同一条真实 planning sample，processor 侧估算过 prompt token 数：

- `1920x1080`：约 `3603`
- `1280x704`：约 `2062`
- `960x540`：约 `1558`
- `640x360`：约 `1211`

因此：

- `960x540` 落在 `2048` 区间内，更容易跑通
- `1280x704` 已经逼近甚至超过 `2048`
- 如果继续大幅抬高 `max-model-len`，当前机器又容易在 vLLM 初始化和 profile 时 OOM

最后形成的折中配置是：

- 图像固定 `1280x704`
- `max-model-len=2560`
- `max-num-batched-tokens=2560`
- `max-num-seqs=1`

### 10.7 `960x540` 语义验证结果

报告：

- `/cache/ma-user/curious_vla_assets/logs/vllm_semantic_validation_20260407_batch6.json`

结果摘要：

- scene 数：`6`
- request 成功：`6/6`
- overall valid：`6/6`
- mean latency：`12.979475s`

这个结果说明：

- 语义 gate 本身成立
- `vllm-ascend` 在当前机器上不是“完全不可用”

### 10.8 `1280x704` 语义验证结果

报告：

- `/cache/ma-user/curious_vla_assets/logs/vllm_semantic_validation_1280x704_20260407.json`

测试 scene：

- `go straight`
- `turn left`

结果摘要：

- request 成功：`2/2`
- overall valid：`2/2`
- mean latency：`15.245859s`

这一步很关键，因为它证明了：

- 当前机器上 `1280x704` 不是理论可行，而是已经在真实 VL planning sample 上实际通过

### 10.9 当前 `1280x704` 可用的服务配置

已验证可工作的服务启动命令：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate /cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend

export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /cache/ma-user/curious_vla_assets/models/Curious-VLA \
  --host 127.0.0.1 \
  --port 18002 \
  --dtype bfloat16 \
  --max-model-len 2560 \
  --max-num-batched-tokens 2560 \
  --max-num-seqs 1 \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

这里最关键的点是：

- `max_new_tokens=512` 不是主要瓶颈
- 真正的瓶颈是多模态 prompt 的总上下文长度
- 为了让 `1280x704` 跑通，需要把 `max-model-len` 提到 `2560`
- 同时把 `max-num-batched-tokens` 和 `max-num-seqs` 压低，避免 profile 阶段 OOM

### 10.10 正式 `1280x704` benchmark 结果

本次正式 benchmark 报告：

- `/cache/ma-user/curious_vla_assets/logs/vllm_planning_latency_benchmark_20260407_1280x704.json`

运行配置：

- base URL：`http://127.0.0.1:18002/v1`
- 图像尺寸：`1280x704`
- warmup runs：`1`
- benchmark runs：`4`
- scene 选择：`diverse-by-command`
- text planning gate：开启

场景类别：

- `go straight`
- `turn left`

结果摘要：

- benchmark count：`4`
- request 成功：`4/4`
- contract valid：`4/4`
- intent alignment valid：`4/4`
- trajectory valid：`4/4`
- overall valid：`4/4`
- recommended for latency benchmark：`true`

request latency 统计：

- mean：`13.188920s`
- p50：`13.200862s`
- p95：`13.371148s`
- min：`12.956358s`
- max：`13.397599s`

客户端侧总场景耗时：

- mean total scene time：`13.262334s`
- p50 total scene time：`13.271470s`
- p95 total scene time：`13.450370s`

客户端除 HTTP request 以外的额外开销：

- mean overhead：`0.073414s`

这说明当前这条 benchmark 里：

- 时间基本都花在服务端 completion 路径
- benchmark 客户端自己的准备和解析开销相对很小

### 10.10.1 更正式的 `5 + 50` benchmark

在前面的 4-run 小规模基线之上，2026 年 4 月 7 日又补跑了一组更正式的 `vllm-ascend` benchmark，用来作为当前更可信的 latency 参考值。

实验参数：

- 运行环境：
  - conda env：`/cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend`
  - 模型目录：`/cache/ma-user/curious_vla_assets/models/Curious-VLA`
  - 模型架构：`Qwen2_5_VLForConditionalGeneration`
  - 模型规模理解：`Qwen2.5-VL-3B` 级别 Curious-VLA
- 图像输入：
  - 数据来源：`warmup_two_stage/sensor_blobs/.../CAM_F0`
  - 原始前视图分辨率：`1920x1080`
  - benchmark 上传分辨率：`1280x704`
  - 图像模态：单张 `front-view` 图像
- 服务配置：
  - `host=127.0.0.1`
  - `port=18002`
  - `dtype=bfloat16`
  - `max-model-len=2560`
  - `max-num-batched-tokens=2560`
  - `max-num-seqs=1`
  - `tensor-parallel-size=1`
- benchmark 配置：
  - `BASE_URL=http://127.0.0.1:18002/v1`
  - `width=1280`
  - `height=704`
  - `scene-limit=4`
  - `selection-mode=diverse-by-command`
  - `warmup-runs=5`
  - `benchmark-runs=50`
  - `max-tokens=512`
  - `temperature=0.0`
  - `run-text-planning-control=true`

实际命令：

```bash
BASE_URL=http://127.0.0.1:18002/v1 \
./local/run_planning_latency_benchmark.sh \
  --backend vllm \
  --scene-limit 4 \
  --selection-mode diverse-by-command \
  --warmup-runs 5 \
  --benchmark-runs 50 \
  --width 1280 \
  --height 704 \
  --max-tokens 512
```

本次 4 个 scene 为：

- `/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/openscene_meta_datas/056e9afeaf8b6c1ca.pkl`
- `/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/openscene_meta_datas/41932cf3f8bc365f0.pkl`
- `/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/openscene_meta_datas/086487f683be38e1b.pkl`
- `/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/openscene_meta_datas/72747cb900d155b4f.pkl`

正式 50 次 benchmark 中：

- `go straight`：`25` 次
- `turn left`：`25` 次

text planning gate 结果：

- `request_ok=True`
- `overall_valid=True`
- latency：`14.771608s`

正式 benchmark 报告：

- `/cache/ma-user/curious_vla_assets/logs/vllm_planning_latency_benchmark_20260407_165915.json`

服务日志：

- `/cache/ma-user/curious_vla_assets/logs/vllm_ascend_2560_formal_20260407.log`

正式 benchmark 汇总结果：

- benchmark count：`50`
- request 成功：`50/50`
- contract valid：`50/50`
- intent alignment valid：`50/50`
- trajectory valid：`50/50`
- overall valid：`50/50`
- recommended for latency benchmark：`true`

request latency 统计：

- mean：`13.098643s`
- p50：`13.036212s`
- p95：`14.298637s`
- min：`12.236058s`
- max：`14.829599s`

总场景耗时统计：

- mean total scene time：`13.169839s`
- p50 total scene time：`13.106756s`
- p95 total scene time：`14.369926s`
- min total scene time：`12.307362s`
- max total scene time：`14.898733s`

客户端额外开销统计：

- mean overhead：`0.071196s`
- p50 overhead：`0.070747s`
- p95 overhead：`0.075561s`

这组结果可以作为当前仓库在 Ascend NPU 上、`vllm-ascend + 1280x704` 配置下更正式的 latency 基线。

### 10.11 同配置下的 `transformers` / `vllm` 直接对比

为了看清两条路径的实际差距，2026 年 4 月 7 日又补跑了一组更直接的对比实验。

对比时固定了这些条件：

- 同一条 scene：
  `086487f683be38e1b.pkl`
- 同一分辨率：
  `1280x704`
- 同一生成上限：
  `512`
- 同样都是：
  `warmup-runs=0`、`benchmark-runs=1`

`transformers` 命令：

```bash
./local/run_planning_latency_benchmark.sh \
  --backend transformers \
  --scene-path /cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/openscene_meta_datas/086487f683be38e1b.pkl \
  --warmup-runs 0 \
  --benchmark-runs 1 \
  --width 1280 \
  --height 704 \
  --max-new-tokens 512
```

结果：

- `validated=True`
- total latency：`325.888s`
- client latency：`325.866s`
- `first_step_error_m ≈ 0.0134`

报告：

- `/cache/ma-user/curious_vla_assets/logs/planning_latency_benchmark_npu_20260407_164257.json`

`vllm` 命令：

```bash
BASE_URL=http://127.0.0.1:18002/v1 \
./local/run_planning_latency_benchmark.sh \
  --backend vllm \
  --scene-path /cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/openscene_meta_datas/086487f683be38e1b.pkl \
  --warmup-runs 0 \
  --benchmark-runs 1 \
  --width 1280 \
  --height 704 \
  --max-tokens 512
```

结果：

- `overall_valid=True`
- request latency：`17.209s`
- total scene time：`17.305s`
- `first_step_error_m ≈ 0.0270`

报告：

- `/cache/ma-user/curious_vla_assets/logs/vllm_planning_latency_benchmark_20260407_165157.json`

如果按总场景耗时对比：

- `transformers`：`325.888s`
- `vllm`：`17.305s`
- 比值约为：`18.83x`

如果按纯推理请求耗时近似理解，结论也接近：

- `transformers client_total_sec`：`325.866s`
- `vllm request_latency_sec`：`17.209s`
- 比值约为：`18.94x`

这组结果说明：

- 在当前这台 Ascend 910B3 上，同一 scene、同一图像分辨率、同一输出长度上限下，`vllm-ascend` 的服务化响应时延明显低于本地 `transformers + torch_npu`
- 两条路径都通过了当前各自的有效性门槛，所以这不是“一个快但无效”的对比
- 但两者仍不是完全同口径：
  - `transformers` 测的是 agent 进程内 planning latency
  - `vllm` 测的是服务化 request / response latency

因此更准确的表述应该是：

- 当前可比实验下，`vllm-ascend` 的单场景规划响应速度大约比本地 `transformers` 快 `19x`
- 但这个数字不能直接当成“模型内核本身快了 19x”

### 10.12 本次对比暴露出的一个环境坑

本次第一次跑 `vllm` benchmark 时，请求并没有真正打到本地服务，而是被全局环境变量里的：

- `BASE_URL=/3de696b9-b2d8-4b55-9618-d69f773f41d9`

覆盖了脚本默认值，导致出现：

- `Invalid URL ... No scheme supplied`

因此当前经验结论是：

- 跑 `vllm` 相关 benchmark 时，最好显式传入
  `BASE_URL=http://127.0.0.1:18002/v1`
- 不要完全依赖 shell 环境里的默认 `BASE_URL`

## 11. 两条 benchmark 的关系

旧的本地 planning benchmark 与新的 `vllm-ascend` benchmark 有关，但不是同一口径。

本地 `transformers` benchmark 更偏向：

- 本地进程内模型推理
- 本地 processor / tokenizer
- `NavsimCoTQwenAgent.compute_trajectory()` 全流程

`vllm-ascend` benchmark 更偏向：

- scene 级输入准备
- OpenAI-compatible 服务 request / response latency
- 返回结果的轻量级语义校验

因此当前更合理的理解是：

- 本地 benchmark：in-process planning latency
- `vllm` benchmark：服务化规划响应 latency

它们都重要，但不能把两组数字直接当成完全同口径结果比较。

## 12. 当前结论

截至 2026 年 4 月 7 日，可以得出的结论是：

- 当前机器上已经能跑通 `transformers + torch_npu` 的 Curious-VLA planning latency
- 当前机器上也已经能跑通 `vllm-ascend` 的 Curious-VLA 服务路径
- `1280x704` 分辨率已经在真实 VL planning sample 上通过语义 gate
- gate 通过后，可以继续做 `vllm` 服务化 latency benchmark
- 在本次同场景、同分辨率、同 token 上限的直接对比里，`vllm-ascend` 的单场景规划响应大约比本地 `transformers` 快 `19x`
- 在现有 warmup 数据上，planning benchmark 已经具备“基本合理性验证”，不再只是纯时延数字

但目前还不能声称：

- 已完成完整 PDM / EPDMS 评测
- 原始 `1920x1080` 服务形态已经长期稳定
- `vllm-ascend` 已完全替代本地 `transformers` benchmark
- 当前 validation 已达到完整规划质量门槛

## 13. 复现入口

### 13.1 model-only latency

```bash
cd /home/ma-user/curious_vla
./local/run_latency_benchmark_npu.sh \
  --warmup-runs 1 \
  --benchmark-runs 2 \
  --max-new-tokens 64
```

### 13.2 本地 `transformers` planning latency

固定 `1280x704`：

```bash
cd /home/ma-user/curious_vla
./local/run_planning_latency_benchmark_npu.sh \
  --warmup-runs 1 \
  --benchmark-runs 1 \
  --max-new-tokens 512
```

原始分辨率：

```bash
cd /home/ma-user/curious_vla
./local/run_planning_latency_benchmark_npu.sh \
  --scene-path /cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/openscene_meta_datas/086487f683be38e1b.pkl \
  --warmup-runs 0 \
  --benchmark-runs 1 \
  --max-new-tokens 512 \
  --use-raw-resolution
```

### 13.3 启动 `vllm-ascend`

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate /cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend

export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /cache/ma-user/curious_vla_assets/models/Curious-VLA \
  --host 127.0.0.1 \
  --port 18002 \
  --dtype bfloat16 \
  --max-model-len 2560 \
  --max-num-batched-tokens 2560 \
  --max-num-seqs 1 \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

### 13.4 跑 `vllm` 语义 gate

```bash
cd /home/ma-user/curious_vla
BASE_URL=http://127.0.0.1:18002/v1 \
WIDTH=1280 \
HEIGHT=704 \
./local/run_vllm_semantic_validation.sh \
  --scene-limit 2 \
  --selection-mode diverse-by-command
```

### 13.5 跑统一 benchmark 入口

`vllm`：

```bash
cd /home/ma-user/curious_vla
BASE_URL=http://127.0.0.1:18002/v1 \
./local/run_planning_latency_benchmark.sh \
  --backend vllm \
  --scene-limit 4 \
  --warmup-runs 1 \
  --benchmark-runs 4
```

`transformers`：

```bash
cd /home/ma-user/curious_vla
./local/run_planning_latency_benchmark.sh \
  --backend transformers \
  --warmup-runs 1 \
  --benchmark-runs 1 \
  --max-new-tokens 512
```

## 14. 已知限制

当前方案还有这些边界：

- 本地 `transformers` planning latency 很慢，`512` token 下单场景可能要数分钟
- `vllm-ascend` 当前更像“服务化响应 benchmark”，不是完整 planner 评测
- warmup 数据每个样本只有 `1` 个 future 点，所以暂时做不了完整 8-step ADE/FDE
- 当前 validation 还不是完整 PDM / EPDMS 门槛，只是基本合理性校验
- 还没有补成完整 `SceneLoader + sequential worker + 全套 PDM eval` 的时延统计
- 当前报告更适合做环境验证和相对对比，不适合直接当最终线上 SLA

## 15. 后续建议

如果后面继续做 NPU 侧 planning latency，建议按这个顺序推进：

1. 保持当前环境版本不变，先做更多 scene 的串行统计。
2. 单独比较 `1280x704` 与 `1920x1080` 对 token 数和生成耗时的影响。
3. 在更多 scene 上汇总：
   - `validated_rate`
   - `validated p50/p95 latency`
   - `mean first-step error`
4. 如果目标是更真实的 planner latency，再补 `SceneLoader` 串行多 scene 路径。
5. 如果目标是部署 latency，再单独补 server 形态，而不是在当前脚本上硬套。
6. 如果要补完整质量指标，需要换到带更完整 future 标注的数据集，而不是只依赖当前 warmup 数据。

## 16. 相关文档

当前 `latency_docs` 目录只保留本文作为主文档。

如果后续需要补充项目总体背景或 NAVSIM 官方说明，可进一步参考：

- [README.md](/home/ma-user/curious_vla/README.md)
- [README.md](/home/ma-user/curious_vla/navsim_eval/README.md)
