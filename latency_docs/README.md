# latency_docs 目录说明

本文用于快速说明 `latency_docs` 目录下每份文档的用途，并区分：

- 哪些更适合作为结果报告阅读
- 哪些更适合作为环境适配手册或执行指南

## 1. 建议优先阅读的结果报告

这几份更偏“结论、现状、结果汇总”，如果你的目标是快速了解当前做到哪一步了，优先看它们。

### [npu_adaptation_summary.md](/home/ma-user/curious_vla/latency_docs/npu_adaptation_summary.md)

用途：

- NPU 适配全过程的大汇总
- 已合并原先独立的 `vllm_ascend_validation.md` 与 `vllm_ascend_1280x704_total_report.md`
- 同时覆盖本地 `transformers` 路径与 `vllm-ascend` 路径
- 涵盖环境、数据、脚本、输入输出含义、语义 gate、benchmark 结果与已知限制

定位：

- 主结果报告
- 也带少量操作说明

### [planning_latency_benchmark_npu.md](/home/ma-user/curious_vla/latency_docs/planning_latency_benchmark_npu.md)

用途：

- 聚焦本地 `transformers + torch_npu` planning latency benchmark
- 说明这条 benchmark 在测什么、如何理解当前结果

定位：

- 小型结果报告

### [latency_benchmark_1280x704.md](/home/ma-user/curious_vla/latency_docs/latency_benchmark_1280x704.md)

用途：

- 解释“当前 benchmark 到底在测什么”
- 帮助区分 latency 与 PDM/EPDMS 这两类不同指标

定位：

- 方法说明 + 结果解释文档

## 2. 更像工具书 / 执行手册的文档

这几份更适合在需要复现环境、下载数据或搭链路时查阅。

### [warmup_benchmark_setup.md](/home/ma-user/curious_vla/latency_docs/warmup_benchmark_setup.md)

用途：

- CUDA/通用环境下跑通 warmup benchmark 的安装与下载总结

定位：

- 执行手册

### [warmup_benchmark_setup_npu.md](/home/ma-user/curious_vla/latency_docs/warmup_benchmark_setup_npu.md)

用途：

- NPU 机器上复现 warmup benchmark 的安装、目录、下载与数据组织指南

定位：

- 执行手册

## 3. 如果只想快速掌握当前状态，推荐阅读顺序

建议顺序：

1. `npu_adaptation_summary.md`
2. `planning_latency_benchmark_npu.md`
3. `latency_benchmark_1280x704.md`

如果你要实际复现，再看：

1. `warmup_benchmark_setup_npu.md`
2. `warmup_benchmark_setup.md`

## 4. 当前目录的推荐理解

如果把 `latency_docs` 分成两类，可以简单理解为：

- 结果报告：
  - `npu_adaptation_summary.md`
  - `planning_latency_benchmark_npu.md`
  - `latency_benchmark_1280x704.md`
- 执行手册：
  - `warmup_benchmark_setup.md`
  - `warmup_benchmark_setup_npu.md`
