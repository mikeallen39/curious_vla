# Curious-VLA 代码导览图 Roadmap

本文面向第一次进入 `/home/ma-user/curious_vla` 的读者，目标不是解释论文细节，而是回答一个更实际的问题：

- 这个仓库的代码到底分成哪几块？
- 如果我只关心推理、评测、latency、训练，各自该从哪里看起？
- 一张图像和一段历史轨迹，最后是怎么变成规划轨迹的？

如果你只想快速抓主线，先看这三处：

1. [navsim_qwen_norm_agent_cot.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py)
2. [curious_vla_client.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/curious_vla_client.py)
3. [run_planning_latency_benchmark.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/run_planning_latency_benchmark.sh)

## 1. 先用一句话理解这个仓库

`curious_vla` 本质上是三层东西叠在一起：

- 顶层项目说明和训练/部署文档
- `navsim_eval` 里的 `NAVSIM` 评测框架 + `Curious-VLA` agent 接入代码
- `EasyR1` 里的 RL 训练代码

而你这段时间主要在跑的 NPU latency，基本发生在：

- `navsim_eval/navsim/agents/curious_vla/`
- `local/`
- `latency_docs/`

## 2. 目录总图

```text
curious_vla/
├── README.md                      # 项目总入口
├── docs/
│   ├── deploy.md                 # 官方部署/评测说明
│   └── train_grpo.md             # GRPO / EasyR1 训练说明
├── navsim_eval/                  # NAVSIM devkit + Curious-VLA agent 接入
│   ├── navsim/
│   │   ├── agents/
│   │   │   └── curious_vla/      # 最核心：Curious-VLA agent 实现
│   │   ├── common/               # Scene / AgentInput / dataloader 等基础数据结构
│   │   └── planning/             # PDM / PDMS / simulator / scoring
│   ├── scripts/evaluation/       # 官方评测脚本
│   ├── docs/                     # NAVSIM 数据与安装说明
│   └── environment.yml           # 官方 navsim 环境定义
├── local/                        # 本地运行脚本，尤其是 NPU / latency 适配
├── EasyR1/                       # RL 训练框架
├── stats/                        # 轨迹标准化统计量
├── assets/                       # 图示资源
└── latency_docs/                 # 当前这轮 NPU / latency 文档
```

## 3. 按“你要做什么”来找代码

### 3.1 如果你要看模型推理主链

优先看：

- [navsim_qwen_norm_agent_cot.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py)
- [curious_vla_client.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/curious_vla_client.py)
- [curious_vla_config.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/curious_vla_config.py)

它们分别负责：

- `navsim_qwen_norm_agent_cot.py`
  - 定义 `NavsimCoTQwenAgent`
  - 决定输入用哪些相机
  - 构造 prompt
  - 调 client 请求模型
  - 把模型输出解析成 8-step trajectory
  - 失败时 fallback 到 constant velocity
- `curious_vla_client.py`
  - 封装对 OpenAI-compatible server 的请求
  - 把本地消息转成 OpenAI chat 格式
  - 负责解析模型输出里的轨迹
- `curious_vla_config.py`
  - 定义 agent 配置项，比如 `model_name_or_path`、`max_tokens`、`temperature`

### 3.2 如果你要看 NAVSIM 官方评测链

优先看：

- [run_qwen_pdm_score_evaluation.sh](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/scripts/evaluation/run_qwen_pdm_score_evaluation.sh)
- [run_pdm_score_one_stage.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py)
- [navsim_qwen_norm_cot_baseline_agent.yaml](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/planning/script/config/common/agent/navsim_qwen_norm_cot_baseline_agent.yaml)

它们分别负责：

- shell 脚本作为评测启动入口
- `run_pdm_score_one_stage.py` 负责：
  - 实例化 agent
  - 加载 scene
  - 调 `compute_trajectory()`
  - 跑 `pdm_score`
  - 汇总成评测结果
- yaml 配置把 hydra 配置系统和 `NavsimCoTQwenAgent` 绑定起来

### 3.3 如果你要看数据如何进入 agent

优先看：

- [dataclasses.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/common/dataclasses.py)
- [dataloader.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/common/dataloader.py)

它们负责：

- `dataclasses.py`
  - 定义 `Camera`、`Cameras`、`EgoStatus`、`AgentInput`、`Scene`
  - 把原始 scene dict 转成 agent 可吃的数据结构
- `dataloader.py`
  - 根据 split/filter 加载 log
  - 构造 `SceneLoader`
  - 通过 token 拿到 `agent_input` 和 `scene`

### 3.4 如果你要看 NPU latency 和本地适配

优先看：

- [run_planning_latency_benchmark.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/run_planning_latency_benchmark.sh)
- [run_planning_latency_benchmark_npu.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_planning_latency_benchmark_npu.py)
- [run_vllm_planning_latency_benchmark.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_vllm_planning_latency_benchmark.py)
- [run_vllm_semantic_validation.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_vllm_semantic_validation.py)

它们分别负责：

- `run_planning_latency_benchmark.sh`
  - 统一入口
  - 用 `--backend transformers|vllm` 切换两条 benchmark 路径
- `run_planning_latency_benchmark_npu.py`
  - 本地 `transformers + torch_npu` 路径
  - 在进程内加载模型
  - 构造一个本地 client 替代远端服务
  - 直接打到 `NavsimCoTQwenAgent.compute_trajectory()`
- `run_vllm_planning_latency_benchmark.py`
  - 走已经启动好的 `vllm-ascend` 服务
  - 测服务 request/response latency
- `run_vllm_semantic_validation.py`
  - 做轻量级合理性验证
  - 检查 contract、意图对齐、轨迹几何合法性

### 3.5 如果你要看训练

优先看：

- [docs/train_grpo.md](https://github.com/mikeallen39/curious_vla/blob/main/docs/train_grpo.md)
- [EasyR1/README.md](https://github.com/mikeallen39/curious_vla/blob/main/EasyR1/README.md)
- [config_vla.yaml](https://github.com/mikeallen39/curious_vla/blob/main/EasyR1/examples/config_vla.yaml)
- [train_qwen_2_5_vl.sh](https://github.com/mikeallen39/curious_vla/blob/main/EasyR1/train_scripts/train_qwen_2_5_vl.sh)

训练相关的逻辑基本不在 `navsim_eval` 主链里，而是在 `EasyR1/`。

## 4. 最核心的一条调用链

这是你最值得记住的主链：

```text
scene pickle / sensor blobs
    -> SceneLoader / AgentInput
    -> NavsimCoTQwenAgent.compute_trajectory()
    -> _build_prompt_messages()
    -> CuriousVLAClient.forward()
    -> OpenAI-compatible API / 本地 transformers
    -> 文本输出
    -> parse trajectory
    -> Trajectory
    -> PDM / latency benchmark
```

把它展开后，就是下面这样。

### 4.1 数据进入

入口在：

- [dataloader.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/common/dataloader.py)
- [dataclasses.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/common/dataclasses.py)

这层负责把：

- `*.pkl` 场景文件
- `sensor_blobs` 下的图片
- ego 历史状态

整理成：

- `AgentInput`
- `Scene`

这里最关键的数据结构是：

- `AgentInput.ego_statuses`
- `AgentInput.cameras`
- `Scene.scene_metadata`

### 4.2 Agent 层

入口在：

- [navsim_qwen_norm_agent_cot.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py)

这个文件是整个项目里最重要的单文件。

你可以把它理解成：

- Curious-VLA 在 NAVSIM 里的“适配器”

它主要做五件事：

1. 根据 `cam_type` 决定传哪种视觉输入
2. 从 `AgentInput` 提取图像、历史轨迹和驾驶指令
3. 拼出多任务 prompt
4. 调 client 请求模型
5. 把返回文本解析为 8 个未来点的轨迹

### 4.3 Prompt 设计

还在：

- [navsim_qwen_norm_agent_cot.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py)

当前 prompt 不是“只输出轨迹”，而是四段任务合在一起：

1. 关键目标/条件检测
2. 自然语言解释
3. meta-behaviour 分类
4. 最终 future trajectory

这也是为什么 Curious-VLA 的默认生成比“只吐 8 个点”更慢。

### 4.4 Client 层

入口在：

- [curious_vla_client.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/curious_vla_client.py)
- [convert_chat_template.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/convert_chat_template.py)

这里的职责是：

- 把 agent 内部消息格式转成 OpenAI chat 接口能接收的格式
- 调 `/v1/chat/completions`
- 从返回文本里解析轨迹

这里支持几类 parser：

- 直接解析二维轨迹
- 解析三维 `(x, y, yaw)` 轨迹
- 从指定字段后解析 trajectory
- 从 action token 解码 trajectory

所以如果后面你想改输出格式，最常改的是这个文件，而不是整个 agent。

### 4.5 评测层

入口在：

- [run_pdm_score_one_stage.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py)

这层做的事情是：

- 根据 hydra 配置实例化 agent
- 按 token 逐场景跑 `compute_trajectory()`
- 把轨迹喂给 `pdm_score`
- 最后产出 `PDMS` 相关结果

所以如果你问：

- “官方 benchmark 从哪一层开始算？”

答案通常是：

- 从 `run_pdm_score_one_stage.py` 开始

而不是从 `local/` 里的 latency 脚本开始。

## 5. 官方主链和本地 NPU 主链的区别

### 5.1 官方主链

```text
scripts/evaluation/run_qwen_pdm_score_evaluation.sh
    -> run_pdm_score_one_stage.py
    -> instantiate agent
    -> SceneLoader
    -> agent.compute_trajectory()
    -> pdm_score
```

这条链回答的是：

- 规划质量好不好
- `PDMS / EPDMS` 怎么样

### 5.2 本地 NPU latency 主链

```text
local/run_planning_latency_benchmark.sh
    -> transformers backend 或 vllm backend
    -> 单场景/少量场景构造输入
    -> agent 或服务调用
    -> 记录 latency / 合理性校验
```

这条链回答的是：

- 在 NPU 上一次请求要多久
- `transformers` 和 `vllm` 差多少
- 输出至少是不是基本合理

所以这两条链相关，但不是同一件事。

## 6. `local/` 目录怎么读

`local/` 可以按四类理解。

### 6.1 环境脚本

- [local_env.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/local_env.sh)
- [local_env_npu.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/local_env_npu.sh)
- [local_env_vllm_ascend.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/local_env_vllm_ascend.sh)

作用：

- 统一目录、环境变量、模型路径、数据路径

### 6.2 数据准备脚本

- [download_model_loop.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/download_model_loop.sh)
- [finalize_warmup_data.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/finalize_warmup_data.sh)
- [finalize_navtest_data.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/finalize_navtest_data.sh)
- [run_metric_caching_warmup.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/run_metric_caching_warmup.sh)
- [run_metric_caching_navtest.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/run_metric_caching_navtest.sh)

作用：

- 下载模型
- 整理数据目录
- 预先构建 metric cache

### 6.3 Benchmark 脚本

- [run_latency_benchmark_npu.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_latency_benchmark_npu.py)
- [run_planning_latency_benchmark_npu.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_planning_latency_benchmark_npu.py)
- [run_vllm_planning_latency_benchmark.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_vllm_planning_latency_benchmark.py)
- [run_vllm_trajectory_only_latency_benchmark.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_vllm_trajectory_only_latency_benchmark.py)

作用：

- 分别测 model-only、planning、service 化、trajectory-only 等不同口径的 latency

### 6.4 运行包装脚本

- [run_planning_latency_benchmark.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/run_planning_latency_benchmark.sh)
- [run_vllm_planning_latency_benchmark.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/run_vllm_planning_latency_benchmark.sh)
- [run_vllm_trajectory_only_latency_benchmark.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/run_vllm_trajectory_only_latency_benchmark.sh)
- [run_warmup_eval.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/run_warmup_eval.sh)
- [run_navtest_eval.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/run_navtest_eval.sh)

作用：

- 帮你少敲环境变量
- 串起 Python 脚本和参数

## 7. `navsim_eval` 目录怎么读

如果你不是做 devkit 开发，只建议重点看四块：

### 7.1 `navsim/agents/curious_vla/`

这是 Curious-VLA 自己的 agent 实现区，也是你最该熟悉的地方。

关键文件：

- [navsim_qwen_norm_agent_cot.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py)
- [curious_vla_client.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/curious_vla_client.py)
- [curious_vla_config.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/curious_vla_config.py)
- [convert_chat_template.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/convert_chat_template.py)

### 7.2 `navsim/common/`

这是数据结构和加载层。

关键文件：

- [dataclasses.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/common/dataclasses.py)
- [dataloader.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/common/dataloader.py)

### 7.3 `navsim/planning/script/`

这是评测入口层。

关键文件：

- [run_pdm_score_one_stage.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py)
- [navsim_qwen_norm_cot_baseline_agent.yaml](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/planning/script/config/common/agent/navsim_qwen_norm_cot_baseline_agent.yaml)

### 7.4 `scripts/evaluation/`

这是官方 shell 启动层。

关键文件：

- [run_qwen_pdm_score_evaluation.sh](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/scripts/evaluation/run_qwen_pdm_score_evaluation.sh)

## 8. `EasyR1` 目录怎么读

如果你当前主要做推理和 latency，可以先不深入 `EasyR1/`。

只有当你开始关心：

- RL 训练怎么做
- ADAS 过滤怎么接
- GRPO 配置在哪里

才需要往下看。

建议顺序：

1. [docs/train_grpo.md](https://github.com/mikeallen39/curious_vla/blob/main/docs/train_grpo.md)
2. [EasyR1/README.md](https://github.com/mikeallen39/curious_vla/blob/main/EasyR1/README.md)
3. [config_vla.yaml](https://github.com/mikeallen39/curious_vla/blob/main/EasyR1/examples/config_vla.yaml)
4. [train_qwen_2_5_vl.sh](https://github.com/mikeallen39/curious_vla/blob/main/EasyR1/train_scripts/train_qwen_2_5_vl.sh)

你可以把 `EasyR1/` 理解成：

- 训练子项目

而把 `navsim_eval/` 理解成：

- 推理评测子项目

## 9. 推荐阅读顺序

### 9.1 如果你是第一次接触 Curious-VLA

建议按这个顺序读：

1. [README.md](https://github.com/mikeallen39/curious_vla/blob/main/README.md)
2. [deploy.md](https://github.com/mikeallen39/curious_vla/blob/main/docs/deploy.md)
3. [navsim_qwen_norm_agent_cot.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py)
4. [curious_vla_client.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/curious_vla_client.py)
5. [dataclasses.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/common/dataclasses.py)
6. [run_qwen_pdm_score_evaluation.sh](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/scripts/evaluation/run_qwen_pdm_score_evaluation.sh)
7. [run_pdm_score_one_stage.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py)

### 9.2 如果你只关心 NPU latency

建议按这个顺序读：

1. [npu_adaptation_summary.md](https://github.com/mikeallen39/curious_vla/blob/main/latency_docs/npu_adaptation_summary.md)
2. [curious_vla_910b_env_repro.md](https://github.com/mikeallen39/curious_vla/blob/main/latency_docs/curious_vla_910b_env_repro.md)
3. [run_planning_latency_benchmark.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/run_planning_latency_benchmark.sh)
4. [run_planning_latency_benchmark_npu.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_planning_latency_benchmark_npu.py)
5. [run_vllm_planning_latency_benchmark.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_vllm_planning_latency_benchmark.py)
6. [run_vllm_semantic_validation.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_vllm_semantic_validation.py)
7. [navsim_qwen_norm_agent_cot.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py)

### 9.3 如果你只关心训练

建议按这个顺序读：

1. [train_grpo.md](https://github.com/mikeallen39/curious_vla/blob/main/docs/train_grpo.md)
2. [EasyR1/README.md](https://github.com/mikeallen39/curious_vla/blob/main/EasyR1/README.md)
3. [config_vla.yaml](https://github.com/mikeallen39/curious_vla/blob/main/EasyR1/examples/config_vla.yaml)
4. [train_qwen_2_5_vl.sh](https://github.com/mikeallen39/curious_vla/blob/main/EasyR1/train_scripts/train_qwen_2_5_vl.sh)

## 10. 你以后最常改哪些文件

如果你后续继续在这个仓库做实验，最常改的通常是：

### 10.1 改 prompt / 改输出 schema

改：

- [navsim_qwen_norm_agent_cot.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py)
- [curious_vla_client.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/curious_vla_client.py)

### 10.2 改 benchmark 口径

改：

- [run_planning_latency_benchmark_npu.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_planning_latency_benchmark_npu.py)
- [run_vllm_planning_latency_benchmark.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_vllm_planning_latency_benchmark.py)
- [run_vllm_trajectory_only_latency_benchmark.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_vllm_trajectory_only_latency_benchmark.py)

### 10.3 改环境与路径

改：

- [local_env_npu.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/local_env_npu.sh)
- [local_env_vllm_ascend.sh](https://github.com/mikeallen39/curious_vla/blob/main/local/local_env_vllm_ascend.sh)

### 10.4 改官方评测配置

改：

- [run_qwen_pdm_score_evaluation.sh](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/scripts/evaluation/run_qwen_pdm_score_evaluation.sh)
- [navsim_qwen_norm_cot_baseline_agent.yaml](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/planning/script/config/common/agent/navsim_qwen_norm_cot_baseline_agent.yaml)

## 11. 最后给一个简化判断

你可以用下面这个规则快速判断一个文件大概属于哪一层：

- 在 `navsim_eval/navsim/agents/curious_vla/`
  - 大概率是 Curious-VLA 业务核心
- 在 `navsim_eval/navsim/common/`
  - 大概率是数据结构和加载逻辑
- 在 `navsim_eval/navsim/planning/script/`
  - 大概率是官方评测入口
- 在 `local/`
  - 大概率是本地实验、NPU 适配、benchmark 包装脚本
- 在 `EasyR1/`
  - 大概率是训练相关
- 在 `latency_docs/`
  - 大概率是本轮实验文档，不是业务代码

## 12. 一页版结论

如果你只记住一件事，就记住这张心智图：

```text
README / docs
    -> 告诉你仓库想做什么

navsim_eval/common
    -> 把数据变成 AgentInput / Scene

navsim_eval/agents/curious_vla
    -> 把 AgentInput 变成 prompt，并调用模型产出轨迹

navsim_eval/planning/script
    -> 把轨迹放进 NAVSIM / PDM 评测

local
    -> 把这整条链拿出来做本地实验、NPU 适配和 latency benchmark

EasyR1
    -> 训练，不是主推理链
```

所以从“理解仓库主干”的角度看，真正最核心的主轴只有一句话：

- `SceneLoader -> AgentInput -> NavsimCoTQwenAgent -> CuriousVLAClient -> trajectory -> PDM / latency`
