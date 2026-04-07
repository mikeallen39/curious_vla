# Curious-VLA NPU 适配总文档

本文汇总当前仓库为 **Ascend NPU** 运行 latency 相关测评所做的全部有效适配，包括：

- 模型与数据下载
- 目录规划
- NPU 环境与关键依赖版本
- benchmark 输入输出含义
- `navsim` 导入链补齐
- 本地 NPU latency benchmark 脚本
- 更接近 end-to-end planning latency 的 benchmark 脚本
- 实测结果
- 已知限制与后续方向

本文描述的是 **当前已经在本机实际跑通** 的方案，不是泛化模板。

## 1. 目标范围

本次适配的目标是：

- 在没有 GPU、只有 NPU 的机器上运行 Curious-VLA latency 测试
- 先跑通模型侧 latency benchmark
- 再补到更接近真实 planning 路径的 benchmark

本次已经实际跑通的两类测试：

- model-only latency
- `AgentInput -> NavsimCoTQwenAgent.compute_trajectory()` planning latency

本次没有继续补到的部分：

- 完整 PDM 质量评测链路
- 全量 `navtest` 传感器数据评测
- OpenAI-compatible server on NPU

## 2. 机器与运行时基线

当前机器环境：

- OS: `EulerOS 2.0 (SP10)`
- NPU: `Ascend 910B3`
- `npu-smi`: `23.0.6`
- CANN: `8.1.RC1`

NPU 可用性通过 `npu-smi info` 已确认。

## 3. 存储与目录规划

由于 `/home/ma-user/work` 剩余空间只有约 `3.2 GB`，不足以放模型和数据，本次所有大文件统一放在：

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
  logs/
  models/
    Curious-VLA/
```

为了兼容仓库原有数据布局，还建立了下面这些软链接形式的入口：

- `/cache/ma-user/curious_vla_assets/data/maps`
- `/cache/ma-user/curious_vla_assets/data/warmup_two_stage`
- `/cache/ma-user/curious_vla_assets/data/navsim_logs/test`

## 4. 已下载资源

当前已就位的资源如下。

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

## 5. 当前可工作的 NPU 环境

当前实际可工作的 conda prefix：

- `/cache/ma-user/curious_vla_assets/envs/curious-vla-npu-latency`

这是本次所有 benchmark 使用的环境。

### 5.1 关键兼容版本

当前确认可工作的关键组合：

- `torch==2.1.0`
- `torch_npu==2.1.0.post12`
- `transformers==4.55.4`
- `numpy==1.26.4`

其他已验证存在的关键包：

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

### 5.2 明确踩过的坑

下面这些结论已经实际验证过：

- `transformers==4.57.1` 与当前 `torch/torch_npu` 组合不兼容
- `transformers==4.55.4` 可以正常加载 `Qwen2_5_VLForConditionalGeneration`
- `pyogrio/rasterio` 相关安装过程曾把 `numpy` 升到 `2.2.6`
- `numpy==2.x` 会破坏当前 `torch_npu`
- 回退到 `numpy==1.26.4` 后环境恢复可用

建议不要再把这套环境随意升级到更高版本的 `transformers` 或 `numpy`。

## 6. navsim / nuplan 导入链适配

为了让 planning latency 能跑到 agent 层，除了模型本身，还补齐了 `navsim` 和 `nuplan` 导入链所需依赖。

当前已确认下面这些导入可用：

- `aioboto3`
- `nuplan`
- `navsim.common.dataclasses`
- `navsim.common.dataloader`
- `navsim.agents.curious_vla.navsim_qwen_norm_agent_cot`

其中最后一个模块最初失败的原因不是 GIS，而是：

- `navsim_qwen_norm_agent_cot.py` 默认读取 `../stats/trajectory_stats_train.json`
- 在当前工作目录下会找不到

因此本次通过环境变量明确指定：

```bash
export STATS_PATH=/home/ma-user/curious_vla/stats/trajectory_stats_train.json
```

## 7. 新增和使用的本地 NPU 脚本

### 7.1 环境入口

新增：

- [local/local_env_npu.sh](/home/ma-user/curious_vla/local/local_env_npu.sh)

作用：

- 统一 `PROJECT_ROOT`
- 统一 asset root
- 指定 conda env prefix
- 指定模型目录
- 指定 warmup 数据目录
- 指定 `STATS_PATH`
- 指定 `OPENSCENE_DATA_ROOT` / `WARMUP_SENSOR_PATH`

### 7.2 model-only latency benchmark

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

这是因为当前 `transformers` 路径会访问这个接口，而 `torch 2.1.0` 上不完整。

### 7.3 planning latency benchmark

新增：

- [local/run_planning_latency_benchmark_npu.py](/home/ma-user/curious_vla/local/run_planning_latency_benchmark_npu.py)
- [local/run_planning_latency_benchmark_npu.sh](/home/ma-user/curious_vla/local/run_planning_latency_benchmark_npu.sh)

这套脚本测的是更接近 end-to-end planning latency 的路径：

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

## 8. benchmark 口径

### 8.1 model-only latency

这条口径只测：

- 模型加载后
- 图像与 prompt tokenization
- `model.generate()`
- decode

不包含：

- `navsim` dataloader
- agent prompt 构造
- `compute_trajectory()` 逻辑

### 8.2 planning latency

这条口径已经比 model-only 更接近真实 planning 路径，包含：

- 真实 warmup scene 读取
- `AgentInput` 构造
- `NavsimCoTQwenAgent._build_prompt_messages`
- agent 内部图像落盘
- 本地 NPU 模型推理
- 输出解析
- 反归一化

但仍然不包含：

- 完整 PDM 质量评测
- 多场景并发 worker
- 真实服务端 HTTP 部署开销

## 9. benchmark 输入输出的实际含义

这一节只解释当前仓库里 **已经在跑的两条 benchmark** 的输入和输出语义。

### 9.1 model-only benchmark 的输入

对应脚本：

- [local/run_latency_benchmark_npu.py](/home/ma-user/curious_vla/local/run_latency_benchmark_npu.py)

这条 benchmark 的输入不是完整场景，而是：

- 一张前视图图片
- 一段固定手写 prompt

脚本里 prompt 的含义是：

- 当前只有 1 张 front-view image
- high-level intent 被写死成 `go straight`
- 历史轨迹也不是从真实 scene 动态构造，而是固定模板

所以这条 benchmark 的真实语义是：

- 测 “单张图 + 固定 prompt” 下，Curious-VLA 模型本体在 NPU 上生成一次要多久

它不是完整 planning 输入。

### 9.2 model-only benchmark 的输出

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

### 9.3 planning benchmark 的输入

对应脚本：

- [local/run_planning_latency_benchmark_npu.py](/home/ma-user/curious_vla/local/run_planning_latency_benchmark_npu.py)

这条 benchmark 的输入来自真实 warmup scene：

- 一个 `openscene_meta_datas/*.pkl`
- 当前每个样本总共 `5` 帧
- 其中前 `4` 帧用于构建 history / current input
- 第 `5` 帧不送进模型，只作为可用真值用于验证

真正喂给模型的信息包括：

- 当前时刻前视图 `cam_f0`
- 过去 4 帧推出来的历史 ego pose
- 当前 high-level driving command
- agent 自动拼出的 CoT prompt

所以它的语义是：

- 测 “真实场景片段输入到 planner agent 后，返回未来轨迹要多久”

### 9.4 planning benchmark 的输出

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

### 9.5 当前 validation 字段的含义

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

## 10. 实测结果

### 10.1 model-only latency

已成功运行命令：

```bash
./local/run_latency_benchmark_npu.sh \
  --warmup-runs 1 \
  --benchmark-runs 2 \
  --max-new-tokens 64
```

结果：

- resize: `1280x704`
- 平均 latency 约 `42.7s`

报告：

- `/cache/ma-user/curious_vla_assets/logs/latency_benchmark_npu_20260406_000347.json`

### 10.2 planning latency，固定 1280x704，非 fallback 早期结果

成功命令：

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

### 10.3 planning latency，原始 1920x1080，非 fallback

成功命令：

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

### 10.4 planning latency，固定 1280x704，加入 validation 后的当前基线

当前最新实际验证命令：

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

这条结果比之前更有意义，因为它不只是“跑完了”，还满足了当前定义的基本合理性门槛。

## 11. token 长度与 fallback 行为

planning benchmark 中最关键的一个现象是：

- `64` new tokens 不够
- `256` new tokens 仍然不够
- `512` new tokens 才在实测 scene 上避免 fallback

原因：

- 当前 CoT prompt 输出非常长
- 模型会先输出 `critical_objects / explanation / meta_behaviour`
- `future_trajectory` 经常在字符串中途被截断
- `NavsimCoTQwenAgent` 解析失败后会退回 constant-velocity fallback

因此当前 planning latency 测试如果想测真实轨迹输出，不建议把 `--max-new-tokens` 设为 `512` 以下。

## 12. 当前适配的关键决策

### 12.1 不走 vLLM

本次没有尝试在 NPU 上继续折腾 `vllm`，而是直接采用：

- `transformers + torch_npu`

原因：

- 这是当前最直接、最容易落地的本地推理路径
- 已经足够支撑 latency benchmark

### 12.2 不先追完整 PDM eval

用户目标是先跑 latency，而不是质量评测，所以优先做的是：

- model-only latency
- closer-to-end-to-end planning latency

而不是：

- 全量 `run_pdm_score_one_stage.py`

### 12.3 用真实 warmup scene，而不是手造假输入

planning benchmark 没有伪造图像和状态，而是直接从真实 `openscene_meta_datas` 读取：

- 真实 history frame
- 真实 `driving_command`
- 真实前视图图像

所以这条 benchmark 比单纯 `generate()` 更接近真实 planning path。

### 12.4 给 planning benchmark 补结果合理性门槛

当前 planning benchmark 已经不再只看 latency，还会同时记录：

- `non_fallback_count`
- `sanity_valid_count`
- `reference_valid_count`
- `validated_count`
- `validated_mean_total_sec`
- `validated_p50_total_sec`
- `validated_p95_total_sec`

这使得 benchmark 可以区分：

- 只是“模型跑出来了”的样本
- “结果也通过当前合理性校验”的样本

目前这套 validation 还不是完整 PDM 质量评测，但已经比纯粹看时延更接近“有效 planning latency”。

## 13. 复现入口

### 13.1 model-only latency

```bash
cd /home/ma-user/curious_vla
./local/run_latency_benchmark_npu.sh --warmup-runs 1 --benchmark-runs 2 --max-new-tokens 64
```

### 13.2 planning latency

固定 `1280x704`：

```bash
cd /home/ma-user/curious_vla
./local/run_planning_latency_benchmark_npu.sh --warmup-runs 1 --benchmark-runs 1 --max-new-tokens 512
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

## 14. 已知限制

当前方案还有这些边界：

- planning latency 很慢，`512` token 下单场景可能要数分钟
- 这条链路仍然是本地 in-process backend，不是 HTTP server latency
- warmup 数据每个样本只有 `1` 个 future 点，所以暂时做不了完整 8-step ADE/FDE
- 当前 validation 还不是完整 PDM/EPDMS 门槛，只是基本合理性校验
- 还没有补成完整 `SceneLoader + sequential worker + 全套 PDM eval` 的时延统计
- 当前报告更多适合做环境验证和相对对比，不适合直接当最终线上 SLA

## 15. 后续建议

如果后面还要继续做 NPU 侧 planning latency，建议按这个顺序推进：

1. 保持当前环境版本不变，先做更多 scene 的串行统计
2. 单独比较 `1280x704` 与 `1920x1080` 对 token 数和生成耗时的影响
3. 在更多 scene 上汇总：
   - `validated_rate`
   - `validated p50/p95 latency`
   - `mean first-step error`
4. 如果目标是更真实的 planner latency，再补 `SceneLoader` 串行多 scene 路径
5. 如果目标是部署 latency，再单独补 server 形态，而不是在当前脚本上硬套

## 16. 相关文档

已有拆分文档仍然保留：

- [docs/warmup_benchmark_setup_npu.md](/home/ma-user/curious_vla/docs/warmup_benchmark_setup_npu.md)
- [docs/latency_benchmark_1280x704.md](/home/ma-user/curious_vla/docs/latency_benchmark_1280x704.md)
- [docs/planning_latency_benchmark_npu.md](/home/ma-user/curious_vla/docs/planning_latency_benchmark_npu.md)

后续如果只看一份，优先看本文。
