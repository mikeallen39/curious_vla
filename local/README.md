# `local/` 目录说明

这个目录主要放本地运行辅助脚本，覆盖环境配置、数据整理、评测、监控和延迟测试等用途。  
其中一部分脚本是通用入口，另一部分则明显绑定特定机器或 Ascend / NPU 环境。

## 环境与路径配置

- `local_env.sh`
  顶层环境入口脚本。会先做设备检测：如果检测到 GPU，则自动 source `gpu/local_env_gpu.sh`；如果检测到 NPU / Ascend，则自动 source `npu/local_env_npu.sh`。也可以通过 `LOCAL_DEVICE_TYPE=gpu` 或 `LOCAL_DEVICE_TYPE=npu` 强制指定。

- `local_env_override.example.sh`
  新机器适配模板。可以复制为 `local_env_override.sh`，单独覆盖项目路径、数据路径、conda 环境路径、模型路径、cache 路径，而不直接修改 `local_env.sh`。

- `gpu/local_env_gpu.sh`
  GPU 机器专用环境变量脚本，当前是写死路径版本。

- `npu/local_env_npu.sh`
  Ascend / NPU 环境专用变量脚本，路径写死为 `/home/ma-user` 和 `/cache/ma-user` 体系，主要给 NPU 延迟测试脚本使用。

- `npu/local_env_vllm_ascend.sh`
  vLLM-Ascend 环境专用变量脚本，同样绑定 Ascend 机器路径，主要给 vLLM-Ascend 的延迟测试和语义校验脚本使用。

## 数据下载与目录整理

- `download_model_loop.sh`
  循环下载 Curious-VLA 模型，直到完整 checkpoint 文件下载齐全。适合网络不稳定时使用。

- `download_http_ranges.py`
  通用分块下载工具。通过 HTTP Range 并发下载大文件不同字节区间，再自动合并。给大体积数据修复脚本复用。

- `download_test_camera_repair.sh`
  专门用于补齐 `test` camera 数据缺失的修复脚本。会按 split 分片下载、解压，只补缺失 log。当前脚本强绑定 `/cache/ma-user/...` 目录结构，属于特定机器工具。

- `finalize_navtest_data.sh`
  把下载下来的 `navtest` 数据整理成项目期望的目录布局，例如 `$DATA_ROOT/navsim_logs/test`、`$DATA_ROOT/sensor_blobs/test`、`$DATA_ROOT/maps`。

- `finalize_warmup_data.sh`
  把下载下来的 `warmup_two_stage` 数据整理成项目期望的目录布局，例如 `$DATA_ROOT/warmup_two_stage`、`$DATA_ROOT/navsim_logs/test`、`$DATA_ROOT/maps`。

## 评测与 Metric Cache

- `eval/run_metric_caching.sh`
  通用 metric caching 入口。通过 `RUN_SPLIT` 控制实际 split，默认是 `warmup_two_stage`。也支持通过 `*_OVERRIDE` 变量覆盖日志路径、传感器路径、synthetic scene 路径和 cache 路径。

- `eval/run_eval.sh`
  通用 one-stage 评测入口。通过 `RUN_SPLIT` 控制实际 split，默认是 `warmup_two_stage`。也支持通过 `*_OVERRIDE` 变量覆盖数据路径和 metric cache 路径。

- `eval/wait_and_run_eval.sh`
  通用串联入口。先等待 metric caching 结束，再等待本地 API 服务可用，最后调用 `run_eval.sh`。通过 `RUN_SPLIT` 控制实际 split。

## 监控与自动化

- `monitor_warmup_eval_progress.sh`
  读取最新评测日志，输出大致进度、活跃进程和当前 GPU 使用情况。适合长时间评测时在终端快速查看状态。

- `watch_and_run_warmup_gpu_smoke.sh`
  监控 `test` camera 数据是否出现 warmup 可用的 original 场景，一旦满足条件就自动挑选 GPU、拉起 smoke 测试服务并跑一版小规模 warmup 验证。这个脚本是为“数据未全量下载完成前先抢跑一版 GPU smoke”准备的。

## LLaMA-Factory 服务启动

- `llama-factory/start_lf_server.sh`
  启动本地 LLaMA-Factory API 服务。默认可直接跑 Hugging Face 推理；如果设置 `infer_backend=vllm`，则会转去调用 `navsim_eval/lf_serve_cot.sh`。

## 延迟测试

- `run_latency_benchmark_npu.py`
  比较底层的 Ascend NPU 模型延迟测试脚本。加载单张图片，构造固定 prompt，做生成并统计时间。

- `run_latency_benchmark_npu.sh`
  `run_latency_benchmark_npu.py` 的 shell 包装脚本，依赖 `local_env_npu.sh`。

- `latency/run_planning_latency_benchmark.sh`
  planning latency 总入口。通过 `--backend` 和 `--device-type` 明确区分“推理后端”和“运行设备”，在脚本内部完成环境加载、conda 激活和参数拼装，然后调用统一 Python 入口。

- `latency/run_planning_latency_benchmark.py`
  统一的 planning latency Python 入口：
  - `--backend hf|transformers|local` 时，走进程内 Transformers + NAVSIM agent 路径；
  - `--backend vllm|lf` 时，走 OpenAI-compatible HTTP 路径；
  - `--device-type gpu|npu` 独立控制设备语义，不再和 backend 混在一起。

- `latency/planning_latency_http_common.py`
  给统一 latency 入口复用的场景选择、prompt 构造、响应解析与轨迹合法性校验辅助模块。它不是独立入口，而是 `latency/` 内部共享逻辑。

## 说明

- 这个目录里有些脚本是历史实验过程中逐步积累的工具，不是完全统一设计后的正式接口。

- 凡是路径写死到 `/home/ma-user` 或 `/cache/ma-user` 的脚本，基本都属于特定 Ascend 机器专用脚本。

- 文件名中带 `warmup` 和 `navtest` 的脚本，不一定严格只服务对应 split。很多只是命名别名，实际行为仍由 `TRAIN_TEST_SPLIT` 决定。
- 现在更推荐直接使用通用入口：
  - `eval/run_metric_caching.sh`
  - `eval/run_eval.sh`
  - `eval/wait_and_run_eval.sh`
  再通过 `RUN_SPLIT=navtest` 或 `RUN_SPLIT=warmup_two_stage` 控制实际 split。

- 如果是在新的 GPU 服务器上使用，通常优先关注这些文件：
  - `local_env.sh`
  - `local_env_override.example.sh`
  - `eval/run_metric_caching.sh`
  - `eval/run_eval.sh`
  - `llama-factory/start_lf_server.sh`
