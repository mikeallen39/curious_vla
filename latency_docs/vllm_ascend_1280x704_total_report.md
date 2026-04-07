# Curious-VLA vLLM-Ascend `1280x704` 总报告

本文汇总当前在 Ascend NPU 机器上，以 `1280x704` 图像分辨率运行
`Curious-VLA + vllm-ascend` 的完整现状，包括：

- 当前可用配置
- 为什么需要这组配置
- 语义 gate 结果
- 正式 benchmark 结果
- 它和本地 `transformers` benchmark 的关系

最后更新时间：

- 2026 年 4 月 7 日

## 1. 这份报告回答什么问题

这份报告主要回答四个问题：

1. 当前 NPU 机器上能不能用 `vllm-ascend` 服务本地 `Curious-VLA`
2. 真实规划 prompt 在 `1280x704` 下能不能跑
3. 跑 latency 之前能不能先过一个基本语义门槛
4. 当前 `1280x704` 配置下的 latency 大概是多少

这份报告不声称：

- 已完成完整 PDM / EPDMS 评测
- 已验证原始 `1920x1080` 的长期稳定服务
- 已和本地 `transformers` 路径做到完全同口径对比

## 2. 机器与资源

当前机器：

- OS：`EulerOS 2.0 (SP10)`
- NPU：`Ascend 910B3`
- CANN：`8.1.RC1`
- 架构：`aarch64`

本次主要使用的路径：

- 模型：
  `/cache/ma-user/curious_vla_assets/models/Curious-VLA`
- warmup 数据：
  `/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage`
- vLLM env：
  `/cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend`
- 日志：
  `/cache/ma-user/curious_vla_assets/logs`

## 3. 当前 `1280x704` 可用的服务配置

之前的 `2048` context 服务配置足够支撑 `960x540`，
但对 `1280x704` 不够稳。

当前已经实际验证可工作的服务启动命令是：

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

- `max_new_tokens=512` 不是瓶颈
- 真正的瓶颈是多模态 prompt 的总上下文长度
- 为了让 `1280x704` 跑通，需要把 `max-model-len` 提到 `2560`
- 同时把 `max-num-batched-tokens` 和 `max-num-seqs` 压低，避免 profile 阶段 OOM

## 4. 为什么 `1280x704` 比 `960x540` 更难

对同一条真实 planning sample，processor 侧估算过 prompt token 数：

- `1920x1080`：约 `3603`
- `1280x704`：约 `2062`
- `960x540`：约 `1558`
- `640x360`：约 `1211`

因此：

- `960x540` 落在 `2048` 区间内，更容易跑通
- `1280x704` 已经逼近甚至超过 `2048`
- 如果继续大幅抬高 `max-model-len`，当前机器又容易在 vLLM 初始化和 profile 时 OOM

所以最后的折中配置就是：

- 图像固定 `1280x704`
- `max-model-len=2560`
- `max-num-batched-tokens=2560`
- `max-num-seqs=1`

## 5. 语义 gate 是什么

为了避免“服务能返回东西，但结果毫无意义”的情况，
本次先补了一套轻量级语义门槛。

入口脚本：

- `run_vllm_semantic_validation.py`
- `run_vllm_semantic_validation.sh`

它包含三层检查：

1. text-only schema control
2. text-only planning-style control
3. 真实 warmup scene + 真实前视图图像的 VL planning 校验

核心检查项包括：

- 严格 JSON 合同
- `critical_objects` 是否完整
- `meta_behaviour` 是否落在合法值域
- scene intent 和输出 `command` 是否对齐
- 轨迹能否被解析
- 反归一化轨迹是否合理
- 和 warmup 唯一 future 点的一步误差是否在阈值内

## 6. 已记录的语义验证结果

### 6.1 `960x540` 小批量验证

报告：

- `/cache/ma-user/curious_vla_assets/logs/vllm_semantic_validation_20260407_batch6.json`

结果摘要：

- scene 数：`6`
- request 成功：`6/6`
- overall valid：`6/6`
- mean latency：`12.979475s`

这个结果说明：

- 语义 gate 本身是成立的
- `vllm-ascend` 在当前机器上并不是完全不可用

### 6.2 `1280x704` 语义验证

报告：

- `/cache/ma-user/curious_vla_assets/logs/vllm_semantic_validation_1280x704_20260407.json`

测试 scene：

- `go straight`
- `turn left`

结果摘要：

- request 成功：`2/2`
- overall valid：`2/2`
- mean latency：`15.245859s`

这说明：

- `1280x704` 在当前配置下已经不只是“理论上可能”
- 而是已经在真实 VL planning sample 上实际验证通过

## 7. 为什么又单独补了一条 benchmark 入口

为了避免把不同后端的语义混到一起，
这次没有继续复用原来的本地 `transformers` benchmark 脚本，
而是单独加了一条 `vllm-ascend` 的 benchmark 入口。

新增文件：

- `local/local_env_vllm_ascend.sh`
- `local/run_vllm_planning_latency_benchmark.py`
- `local/run_vllm_planning_latency_benchmark.sh`
- `local/run_planning_latency_benchmark.sh`

其中统一入口是：

- `local/run_planning_latency_benchmark.sh`

它支持：

- `--backend transformers`
- `--backend vllm`

这样现在切换后端不需要手动换脚本名。

## 8. 正式 `1280x704` benchmark 结果

本次正式 benchmark 报告文件：

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

### 8.1 gate 结果

- text planning control：通过

### 8.2 benchmark 汇总

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

benchmark 客户端侧的总场景耗时：

- mean total scene time：`13.262334s`
- p50 total scene time：`13.271470s`
- p95 total scene time：`13.450370s`

客户端除 HTTP request 以外的额外开销：

- mean overhead：`0.073414s`

这说明当前这条 benchmark 里：

- 时间基本都花在服务端 completion 路径
- benchmark 客户端自己的准备和解析开销相对很小

## 9. 和旧的 `transformers` benchmark 有什么关系

旧的本地 planning benchmark 在 `transformers + torch_npu` 下测到过：

- 固定 `1280x704`，1 条真实 scene，大约 `346.97s`

但它测的不是一回事。

旧 benchmark 更偏向：

- 本地进程内模型推理
- 本地 processor / tokenizer
- `NavsimCoTQwenAgent.compute_trajectory()` 全流程

新的 `vllm-ascend` benchmark 更偏向：

- scene 级输入准备
- OpenAI 兼容服务 request / response latency
- 返回结果的轻量级语义校验

因此当前最合理的理解是：

- 旧 benchmark：本地 in-process planning latency
- 新 benchmark：服务化规划响应 latency

它们相关，但不能直接把数字当成完全同口径结果对比。

## 10. 当前结论

截至 2026 年 4 月 7 日，可以得出的结论是：

- 当前机器上已经能以 `1280x704` 跑通 `Curious-VLA + vllm-ascend`
- 跑前可以先过一个轻量级 planning 语义门槛
- 在 gate 通过后，可以继续做服务化 latency benchmark

但还不能说：

- 已经完成完整 PDM / EPDMS 验证
- 原始 `1920x1080` 服务形态长期稳定
- 它已经完全替代本地 `transformers` planning benchmark

## 11. 当前推荐使用方式

当前最推荐的顺序是：

1. 按已验证配置启动 `vllm-ascend`
2. 先跑语义 gate
3. gate 通过后再跑 `vllm` latency benchmark

这样得到的时延结果，比单纯看“服务能不能返回点东西”要更有意义。

## 12. 推荐命令

### 12.1 启动服务

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

### 12.2 跑语义 gate

```bash
BASE_URL=http://127.0.0.1:18002/v1 \
WIDTH=1280 \
HEIGHT=704 \
./local/run_vllm_semantic_validation.sh \
  --scene-limit 2 \
  --selection-mode diverse-by-command
```

### 12.3 跑 benchmark

```bash
./local/run_planning_latency_benchmark.sh \
  --backend vllm \
  --scene-limit 4 \
  --warmup-runs 1 \
  --benchmark-runs 4
```

### 12.4 切回本地 `transformers` benchmark

```bash
./local/run_planning_latency_benchmark.sh \
  --backend transformers \
  --warmup-runs 1 \
  --benchmark-runs 1 \
  --max-new-tokens 512
```
