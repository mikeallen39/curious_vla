# Curious-VLA Latency Benchmark 说明（1280x704）

本文说明当前 Curious-VLA 在本仓库中的 benchmark 链路到底测了什么，以及如果目标改成：

- 以 `latency` 作为衡量标准
- 图片分辨率固定为 `1280x704`

应该如何定义、修改和执行。

## 1. 先说结论

当前正在跑的这条 benchmark：

- **不是 latency benchmark**
- 默认衡量的是 **规划质量/性能指标**
- 主要输出是：
  - `PDMS`
  - `EPDMS`

对应代码在：

- [run_pdm_score_one_stage.py](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py#L297)
- [run_pdm_score_one_stage.py](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py#L335)

从实现上看，这条链路会：

- 对多个场景并发评测
- 调模型服务做规划
- 最后输出每个 token 的规划结果和平均质量分数

它不会默认输出：

- 单请求 latency
- 平均 latency
- `p50/p95` latency

## 2. 当前 benchmark 实际在测什么

当前 warmup benchmark 的核心入口是：

- [run_warmup_eval.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/run_warmup_eval.sh)
- [run_navtest_eval.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/run_navtest_eval.sh)
- [run_pdm_score_one_stage.py](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py)

在这个脚本里，评测结果最终会写成 CSV，并输出：

- `pdms_v1`
- `epdms_v2`

实现位置：

- [run_pdm_score_one_stage.py](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py#L297)
- [run_pdm_score_one_stage.py](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py#L300)

日志里打印的总结也是：

- successful scenarios
- failed scenarios
- average PDMS
- average EPDMS

而不是 latency。

## 3. 为什么当前链路不能直接拿来测 latency

原因有三类。

### 3.1 输入分辨率不符合目标

当前代理侧会直接把相机原图保存后发送：

- [navsim_qwen_norm_agent_cot.py](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py#L234)

当前 warmup 数据里的前视图原始分辨率实际是：

- `1920x1080`

也就是说：

- 现在不是固定 `1280x704`
- 而是先送原图，再由后端内部继续处理

### 3.2 服务端像素上限和 `1280x704` 冲突

当前服务端启动脚本里写的是：

- `--image_max_pixels 262144`

对应位置：

- [start_lf_server_gpu1.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/start_lf_server_gpu1.sh#L22)
- [start_lf_server.sh](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/local/start_lf_server.sh#L26)

但：

- `1280 * 704 = 901120`

所以如果你目标是固定 `1280x704`，当前这条服务链路会把图片进一步压小。  
这意味着现在测到的不是“1280x704 下的时延”。

## 3.3 当前 benchmark 是并发评测，不是单流 latency

当前 `run_pdm_score_one_stage.py` 会按 token 分片，再通过 worker 并发执行：

- [run_pdm_score_one_stage.py](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py#L280)
- [run_pdm_score_one_stage.py](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py#L287)

默认使用的是并发 worker：

- [single_machine_thread_pool.yaml](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/planning/script/config/common/worker/single_machine_thread_pool.yaml)

这更接近：

- 并发场景下的整体评测
- 带队列/竞争/进程调度影响的响应时间

而不是：

- 单请求 latency benchmark

## 4. 如果目标是 latency，需要先定义口径

建议先明确你要测哪一种 latency。

### 4.1 API latency

定义：

- 只统计一次模型请求从发出到收到响应的时间

最合适的计时点：

- `client.chat.completions.create(...)` 前后

对应代码位置：

- [curious_vla_client.py](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/agents/curious_vla/curious_vla_client.py#L246)

这个指标更接近：

- 模型服务推理延迟

### 4.2 End-to-end planning latency

定义：

- 从 `compute_trajectory()` 进入，到轨迹返回为止的时间

对应代码位置：

- [navsim_qwen_norm_agent_cot.py](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py#L208)

这个指标包含：

- 图片保存
- base64 编码
- HTTP 请求
- 模型推理
- 响应解析
- 轨迹反归一化

这个更接近：

- 真实规划链路延迟

## 5. 如果要求是 `1280x704` latency，必须做的改动

至少要改三类内容。

### 5.1 在代理侧强制 resize 到 `1280x704`

现在图片是直接原图保存的：

- [navsim_qwen_norm_agent_cot.py](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py#L238)

要满足你的口径，必须改成：

- 先把输入图像 resize 到 `1280x704`
- 再保存
- 再发给模型

否则测到的仍然不是固定分辨率 latency。

### 5.2 把服务端 `image_max_pixels` 提到至少 `901120`

当前是：

- `262144`

但 `1280x704` 需要至少：

- `901120`

所以服务端至少要改成：

```bash
--image_max_pixels 901120
```

否则后端还会继续把图压小。

### 5.3 在客户端或代理中显式记录 latency

当前代码只做调用，不记录 latency：

- [curious_vla_client.py](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/agents/curious_vla/curious_vla_client.py#L246)

需要在这里补：

- `time.perf_counter()` 开始时间
- `time.perf_counter()` 结束时间
- 把 latency 写入日志或结果文件

## 6. 推荐的测法

### 6.1 测单请求 API latency

如果你的目标是模型服务时延，推荐：

- 固定一张 `1280x704` 图片
- 固定 prompt
- 串行请求
- 先 warmup 若干次
- 再统计正式样本

建议输出：

- `avg`
- `p50`
- `p95`
- `min`
- `max`
- 样本数 `N`

### 6.2 测端到端 planning latency

如果你的目标是规划链路时延，推荐：

- 仍然固定 `1280x704`
- 仍然先 warmup
- 但 benchmark 时改成串行执行

当前仓库里可以直接用的串行 worker 是：

- [sequential.yaml](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/planning/script/config/common/worker/sequential.yaml)

也就是说，测时延时不建议继续用现在的：

- `worker=single_machine_thread_pool`

而应改成：

- `worker=sequential`

否则并发会污染 latency。

## 7. 不建议怎样测

下面这些方式不建议直接作为 latency benchmark：

- 直接沿用当前正在跑的 warmup benchmark 结果
- 不改图片分辨率，直接看现有耗时
- 保持 `--image_max_pixels 262144`
- 使用多 worker 并发结果来代表单请求 latency

这些做法测出来的更像：

- 当前实现下的综合运行时间
- 或者并发条件下的响应时间

而不是你要求的：

- `1280x704` 条件下的 latency

## 8. 建议的最终口径

如果你要对外汇报，建议写成下面这种形式：

- 输入分辨率：`1280x704`
- 模式：single-image, single-request, single-worker
- 时延定义：API latency / end-to-end planning latency
- warmup 次数：例如 `10`
- 正式统计样本数：例如 `100`
- 报告指标：`avg / p50 / p95 / min / max`

## 9. 当前状态结论

当前这台机器上正在跑的 benchmark：

- **是规划质量 benchmark**
- **不是 latency benchmark**

当前默认口径下，能直接拿到的是：

- `PDMS`
- `EPDMS`

当前默认口径下，不能直接代表你要求的是：

- `1280x704` 下的 latency

## 10. 下一步建议

如果后面你要真的开始测 latency，推荐按这个顺序改：

1. 代理侧把图片强制 resize 到 `1280x704`
2. 服务端把 `image_max_pixels` 提到 `901120`
3. 在客户端补 API latency 计时
4. 在代理侧补 end-to-end latency 计时
5. 新增一个串行 latency 脚本
6. 最后再跑小样本统计 `avg/p50/p95`

如果需要，我可以直接继续把这一套改成可执行版本，包括：

- 固定 `1280x704` 的图像预处理
- latency 日志记录
- 串行 latency benchmark 脚本
- 自动汇总 `avg/p50/p95`
