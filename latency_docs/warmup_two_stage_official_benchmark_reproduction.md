# Curious-VLA `warmup_two_stage` 官方 Benchmark 复现记录

本文记录 2026-04-09 在 Ascend 910B 环境下，对 Curious-VLA 进行一次正式 `warmup_two_stage` 官方 benchmark 复现的过程、配置、结果与口径说明。

## 1. 本次复现的结论

本次已经成功跑通官方 `warmup_two_stage` 评测链路，并产出正式分数：

- `PDMS`: `0.3738928365083659`
- `EPDMS(V2)`: `0.3225864923840719`
- 成功场景数：`220 / 220`
- 失败场景数：`0`

结果文件：

- `../exp_root/curious_vla_warmup_eval_retry/2026.04.09.08.32.12/2026.04.09.09.15.22.csv`

评测日志：

- `/cache/ma-user/curious_vla_assets/logs/warmup_pdm_eval_retry_20260409_083158.log`

这个结果的意义是：

- 已经不是之前那种自定义 latency proxy
- 而是走了 NAVSIM 官方 `run_pdm_score_one_stage.py` 的正式评测链路
- 因而可以视为一组“正式 benchmark 分数”

但也要明确：

- 这组分数对应的是 `warmup_two_stage`
- 它是公开、较小、用于 warmup leaderboard 对齐的 split
- 不是最终 challenge 私有评测集

## 2. 本次运行配置

### 2.1 机器与环境

- OS：`EulerOS 2.0 (SP10)`
- NPU：`Ascend 910B3`
- CANN：`8.1.RC1`
- benchmark Python 环境：
  `/cache/ma-user/curious_vla_assets/envs/curious-vla-npu-latency`
- 推理服务环境：
  `/cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend`

### 2.2 模型

- 模型目录：
  `/cache/ma-user/curious_vla_assets/models/Curious-VLA`
- 模型来源：
  `MashiroLn/Curious-VLA`
- 项目 README 标注基座：
  `Qwen2.5-VL-3B-Instruct`

### 2.3 数据

本次评测实际使用的数据如下：

- 地图：
  `/cache/ma-user/curious_vla_assets/data/downloads/maps`
- `test` metadata：
  `/cache/ma-user/curious_vla_assets/data/downloads/test_navsim_logs/test`
- `warmup_two_stage` 传感器与 synthetic scenes：
  `/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage`

其中：

- `navsim_log_path` 指向 `test` metadata
- `original_sensor_path` / `synthetic_sensor_path` / `synthetic_scenes_path` 指向 `warmup_two_stage`

这是当前这条 two-stage warmup 评测应采用的正确口径。

### 2.4 vLLM 服务配置

本次 benchmark 使用的是 `vllm-ascend` 服务，而不是进程内 `transformers`：

```bash
vllm serve /cache/ma-user/curious_vla_assets/models/Curious-VLA \
  --host 127.0.0.1 \
  --port 8192 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 2560 \
  --max-num-batched-tokens 2560 \
  --max-num-seqs 1 \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":262144}'
```

服务地址：

- `http://127.0.0.1:8192/v1`

### 2.5 正式评测命令

本次成功跑通的正式命令如下：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export NUPLAN_MAP_VERSION='nuplan-maps-v1.0'
export NUPLAN_MAPS_ROOT='/cache/ma-user/curious_vla_assets/data/downloads/maps'
export NAVSIM_DEVKIT_ROOT='/home/ma-user/curious_vla/navsim_eval'
export NAVSIM_EXP_ROOT='/home/ma-user/curious_vla/exp_root'
export OPENSCENE_DATA_ROOT='/cache/ma-user/curious_vla_assets/data'
export STATS_PATH='/home/ma-user/curious_vla/stats/trajectory_stats_train.json'

/cache/ma-user/curious_vla_assets/envs/curious-vla-npu-latency/bin/python \
  /home/ma-user/curious_vla/navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py \
  train_test_split=warmup_two_stage \
  experiment_name=curious_vla_warmup_eval_retry \
  agent=navsim_qwen_norm_cot_baseline_agent \
  agent.config.model_name_or_path=/cache/ma-user/curious_vla_assets/models/Curious-VLA \
  +agent.config.api_base_url=http://127.0.0.1:8192/v1 \
  +agent.config.max_tokens=512 \
  +agent.config.temperature=0.0 \
  navsim_log_path=/cache/ma-user/curious_vla_assets/data/downloads/test_navsim_logs/test \
  original_sensor_path=/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/sensor_blobs \
  synthetic_sensor_path=/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/sensor_blobs \
  synthetic_scenes_path=/cache/ma-user/curious_vla_assets/data/downloads/warmup_two_stage/synthetic_scene_pickles \
  metric_cache_path=/home/ma-user/curious_vla/exp_root/metric_cache_warmup_two_stage_eval \
  worker=sequential
```

### 2.6 metric cache

本次正式评测复用了已经成功生成的 metric cache：

- `/home/ma-user/curious_vla/exp_root/metric_cache_warmup_two_stage_eval`

对应日志里已经确认：

- `All 220 features and targets were cached successfully`

## 3. 为跑通正式 benchmark 做的必要适配

这次能跑通，不只是“命令对了”，还依赖了几处代码适配。

### 3.1 让 client 真正使用配置里的服务参数

文件：

- `../navsim_eval/navsim/agents/curious_vla/curious_vla_client.py`
- `../navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py`

核心修改：

- agent 初始化时，把 `api_base_url`、`api_key`、`max_tokens`、`temperature` 传给 `CuriousVLAClient`
- client 发请求时不再硬编码 `max_tokens=4096`
- `OpenAI(api_key=...)` 不再硬编码固定值

如果不改这里，当前服务会因为上下文长度限制直接报错。

### 3.2 缺失前视图时自动 fallback

文件：

- `../navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py`

问题：

- 少数样本的 `cam_f0` 是空的
- 原逻辑会先访问 `agent_input.cameras[-1].cam_f0.image`
- 于是直接触发 `AttributeError`

现在的处理是：

- 如果前视图缺失，直接回退到 constant-velocity fallback
- 不再让单个样本把整轮评测打断

### 3.3 没有相邻原始帧映射时，跳过 two-frame 聚合

文件：

- `../navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py`

问题：

- 在当前 `warmup_two_stage` 口径下，没有可用的相邻原始帧映射
- 原脚本依然默认要做 `two_frame_extended_comfort` 聚合
- 最后会在 `pd.concat(all_updates)` 这里报 `ValueError: No objects to concatenate`

现在的处理是：

- 如果 `all_mappings` 为空，就跳过这一步聚合
- 保留本轮 one-stage 结果并完成收尾导出

这一步非常关键，因为第一次完整跑到 `220 / 220` 之后，就是死在这里。

## 4. 结果口径怎么理解

### 4.1 这次结果是不是正式 benchmark

是。

原因：

- 用的是官方 `run_pdm_score_one_stage.py`
- 用的是正式 metric cache
- 用的是官方 `warmup_two_stage` split
- 最终导出了标准 CSV，并统计了 `PDMS` / `EPDMS`

所以它已经是正式 benchmark 分数，不是单纯的 latency 实验结果。

### 4.2 这次结果是不是论文主表结果

不一定。

当前只能明确说：

- 这次结果是 `warmup_two_stage` 口径
- 它应该与 Hugging Face warmup leaderboard 的本地复现口径一致

根据 NAVSIM 官方文档：

- `warmup_two_stage` 是“小型公开 warmup 测试集”
- 本地跑出的结果应当和 warmup leaderboard 上看到的结果对齐

但论文 / README 里展示的高分结果，不一定就是这个 split。

## 5. 能不能用论文中的数据

可以分三种情况看。

### 5.1 如果你指的是“公开且更正式的本地 NAVSIM v2 数据”

可以，用：

- `navhard_two_stage`

这是 NAVSIM 官方文档里定义的：

- “用于本地测试 agent 在 NAVSIM v2 two-stage pseudo closed-loop 设置下表现的标准 split”

相比当前的 `warmup_two_stage`：

- 更正式
- 数据量更大
- 更适合作为本地 benchmark

但当前机器上还没有完整下载它，所以这次没有直接跑。

### 5.2 如果你指的是“warmup leaderboard 对应公开数据”

可以，而且这次已经用了。

当前这次跑的就是：

- `warmup_two_stage`

它本来就是官方提供给 warmup leaderboard 对齐使用的公开数据。

### 5.3 如果你指的是“论文最终 challenge / 主榜数据”

通常不能完整在本地直接复现。

原因是 NAVSIM 官方文档明确写了：

- `private_test_hard_two_stage` 是 challenge leaderboard 使用的私有测试集
- 该数据不公开标注
- 最终分数需要通过提交 `submission.pkl` 到 Hugging Face leaderboard 才能得到

所以：

- 想做“更正式的本地复现”，下一步应下载并跑 `navhard_two_stage`
- 想做“和最终 challenge 榜单同口径”的复现，只能走 submission 流程，不能单靠本地脚本拿到完全同口径的最终分数

## 6. 下一步建议

如果目标是逐步逼近论文口径，推荐顺序如下：

1. 保留当前这次 `warmup_two_stage` 成果，作为“官方链路已跑通”的基线。
2. 下载 `navhard_two_stage`，在同一套代码上再跑一轮正式 benchmark。
3. 如果需要对齐最终 challenge / 论文主榜，再研究生成 `submission.pkl` 并提交到对应 leaderboard。

如果只是想先获得一个“比 warmup 更正式、但仍可本地复现”的结果，那么下一步最值得做的是：

- 跑 `navhard_two_stage`

