# Ascend NPU 上的 Planning Latency Benchmark

本文说明当前仓库中基于 `transformers + torch_npu` 的
planning latency benchmark 在测什么、如何运行，以及当前已经记录到的
实测结果。

## 1. 这条 benchmark 测的是什么

这条 benchmark 比单纯的 model-only latency 更接近真实规划链路，它会：

- 从 `openscene_meta_datas` 里读取真实 warmup scene pickle
- 用真实历史帧构造 `AgentInput`
- 调用 `NavsimCoTQwenAgent.compute_trajectory()`
- 使用本地进程内的 `transformers + torch_npu` 后端推理
- 统计从 scene 输入到轨迹返回的总耗时

它不是：

- OpenAI 兼容服务的 HTTP latency
- vLLM 服务端 latency
- 完整 PDM / EPDMS 评测耗时

## 2. 运行环境

这条 benchmark 使用的环境和资源由下面这个脚本统一配置：

- `local/local_env_npu.sh`

主要路径如下：

- 模型目录：
  `/cache/ma-user/curious_vla_assets/models/Curious-VLA`
- warmup 数据：
  `/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage`
- conda 环境：
  `/cache/ma-user/curious_vla_assets/envs/curious-vla-npu-latency`

## 3. 运行命令

### 3.1 固定缩放到 `1280x704`

这条命令更接近之前的 `1280x704` 基线：

```bash
./local/run_planning_latency_benchmark_npu.sh \
  --warmup-runs 1 \
  --benchmark-runs 1 \
  --max-new-tokens 512
```

### 3.2 使用前视图原始分辨率

这条命令更接近真实 planning 输入，但耗时会更长：

```bash
./local/run_planning_latency_benchmark_npu.sh \
  --scene-path /cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/openscene_meta_datas/086487f683be38e1b.pkl \
  --warmup-runs 0 \
  --benchmark-runs 1 \
  --max-new-tokens 512 \
  --use-raw-resolution
```

## 4. 参数说明

### `max-new-tokens`

当前 CoT prompt 下，输出是比较长的严格 JSON + 轨迹文本。

已经实际验证过：

- `64` 或 `256` 个新 token 不够，容易截断输出
- 输出一旦被截断，轨迹解析就会失败
- 解析失败后 agent 会 fallback 到 constant-velocity

所以在当前测试场景下：

- `512` 是一个比较稳妥的下限

### `agent_overhead_sec`

报告里的 `agent_overhead_sec` 不是模型 forward 自身，而是：

```text
compute_trajectory 总耗时 - 本地后端 forward 总耗时
```

它大致反映：

- prompt 组织
- 输入准备
- agent 层封装
- 轨迹解析

这些非模型 forward 的额外时间。

## 5. 当前已记录结果

基于 2026 年 4 月 6 日当前机器上的实测结果：

- 固定 `1280x704`，真实 benchmark scene 1 条，`max-new-tokens=512`
  - 约 `346.97s`
- 原始 `1920x1080`，真实 benchmark scene 1 条，`max-new-tokens=512`
  - 约 `611.01s`

对应报告文件：

- `/cache/ma-user/curious_vla_assets/logs/planning_latency_benchmark_npu_20260406_100458.json`
- `/cache/ma-user/curious_vla_assets/logs/planning_latency_benchmark_npu_20260406_101232.json`

## 6. 如何理解这条 benchmark

这条 benchmark 最适合回答的问题是：

- 在当前 NPU 机器上，使用本地 `transformers + torch_npu` 路径时，
  更接近真实 planning 的耗时大概是多少？

它不适合直接回答：

- 服务端 API latency 是多少？
- vLLM 路径是否更快？
- 完整评测链路的总耗时是多少？

如果你的目标变成：

- 比较 `transformers` 和 `vllm` 两种后端
- 以服务化请求 latency 为主要指标

那么应该看：

- `vllm_ascend_validation.md`
- `vllm_ascend_1280x704_total_report.md`

## 7. 建议

如果你只是想继续跟踪本地 planning latency，建议优先：

1. 固定 `1280x704`
2. 固定 `max-new-tokens=512`
3. 增加 benchmark scene 数量
4. 同时看 `validated_count`，不要只看时延

如果你要切换成统一入口，可以使用：

```bash
./local/run_planning_latency_benchmark.sh --backend transformers ...
```
