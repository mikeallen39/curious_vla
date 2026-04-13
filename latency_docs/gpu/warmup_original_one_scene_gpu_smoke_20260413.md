# Warmup Original 单场景 GPU Smoke 记录

## 1. 目的

这次测试的目标是：

- 不等 `test` 全量数据下载完成
- 只要 `warmup_two_stage` 相关的 7 个 log 中任意一个在 `test camera` 原始图像里可用
- 就立刻跑一版最小规模 GPU 评测
- 优先确认 `original scene` 在 GPU 上是否真的能走通

这份文档记录的是 2026-04-13 这一轮已经成功完成的 GPU smoke，可作为后续 NPU 对照样例。

## 2. 测试环境

项目根目录：

- `/mnt/42_store/zxz/HUAWEI/VLA/curious_vla`

数据根目录：

- `/data/zxz/HUAWEI/VLA/navsim_data`

模型目录：

- `/data/zxz/condaenv/curious_vla/models/Curious-VLA`

评测环境：

- `/data/zxz/condaenv/curious_vla/navsim`

服务环境：

- `/data/zxz/condaenv/curious_vla/lf`

这次实际使用的是：

- GPU `3`
- 端口 `8192`
- `llamafactory-cli api`
- `infer_backend=huggingface`

## 3. 这次样例里两个容易混淆的 token

这轮实验里要区分两个 token。

### 3.1 探针 token

我用来确认“这个 log 的 original front camera 已经下载到本地”的探针 token 是：

- `51ad5207706e5602`

它对应：

- `log_name = 2021.09.16.19.27.01_veh-45_01749_03230`
- `frame_index = 3`
- `scene_token = e6f686e445f7519d`
- `CAM_F0 = 2021.09.16.19.27.01_veh-45_01749_03230/CAM_F0/00592dcfd4945771.jpg`

它只用于“确认 original 图像已经存在”，不是最后 CSV 里被评分的 token。

### 3.2 真正被评分的 token

这轮 warmup split 实际被 evaluator 选中并评分的 token 是：

- `19e90f2757b25f38`

它同样属于同一个 log：

- `log_name = 2021.09.16.19.27.01_veh-45_01749_03230`
- `frame_index = 1085`
- `scene_token = 945ef2ca37f257fd`
- `CAM_F0 = 2021.09.16.19.27.01_veh-45_01749_03230/CAM_F0/fcad8a4b5334546a.jpg`

最终评分 CSV 里的 token 也是这个：

- `19e90f2757b25f38`

## 4. 实际使用的数据路径

这轮 one-scene eval 使用的路径是：

- `navsim_log_path=/data/zxz/HUAWEI/VLA/navsim_data/navsim_logs/test`
- `original_sensor_path=/data/zxz/HUAWEI/VLA/navsim_data/downloads/openscene-v1.1/sensor_blobs/test`
- `synthetic_sensor_path=/data/zxz/HUAWEI/VLA/navsim_data/warmup_two_stage/sensor_blobs`
- `synthetic_scenes_path=/data/zxz/HUAWEI/VLA/navsim_data/warmup_two_stage/synthetic_scene_pickles`

注意：

- 这次评的是 `original scene`
- 所以关键是 `original_sensor_path` 不能指到 `warmup_two_stage/sensor_blobs`
- 必须指向 `test` 的原始相机数据

## 5. 实际使用的评测约束

这轮 smoke 只评 1 个 log，且只评 original scene：

- `train_test_split=warmup_two_stage`
- `train_test_split.scene_filter.log_names=['2021.09.16.19.27.01_veh-45_01749_03230']`
- `train_test_split.scene_filter.max_scenes=1`
- `train_test_split.scene_filter.include_synthetic_scenes=false`
- `worker=sequential`

## 6. 这次 GPU 服务的启动方式

这轮服务是 watcher 自动起的，本质上等价于：

```bash
source /home/zxz/anaconda3/etc/profile.d/conda.sh
conda activate /data/zxz/condaenv/curious_vla/lf
cd /mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval

CUDA_VISIBLE_DEVICES=3 \
API_HOST=127.0.0.1 \
API_PORT=8192 \
API_VERBOSE=0 \
llamafactory-cli api \
  --model_name_or_path /data/zxz/condaenv/curious_vla/models/Curious-VLA \
  --template qwen2_vl \
  --infer_backend huggingface \
  --image_max_pixels 262144 \
  --trust_remote_code true
```

服务日志：

- [warmup_gpu_watch_server_20260413_075936.log](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/exp_root/warmup_gpu_watch_server_20260413_075936.log)

## 7. 这次实际执行的评测流程

### 7.1 metric cache

这轮先构建了单场景 metric cache：

```bash
/data/zxz/condaenv/curious_vla/navsim/bin/python \
  /mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/planning/script/run_metric_caching.py \
  train_test_split=warmup_two_stage \
  "train_test_split.scene_filter.log_names=['2021.09.16.19.27.01_veh-45_01749_03230']" \
  train_test_split.scene_filter.max_scenes=1 \
  train_test_split.scene_filter.include_synthetic_scenes=false \
  navsim_log_path=/data/zxz/HUAWEI/VLA/navsim_data/navsim_logs/test \
  original_sensor_path=/data/zxz/HUAWEI/VLA/navsim_data/downloads/openscene-v1.1/sensor_blobs/test \
  synthetic_sensor_path=/data/zxz/HUAWEI/VLA/navsim_data/warmup_two_stage/sensor_blobs \
  synthetic_scenes_path=/data/zxz/HUAWEI/VLA/navsim_data/warmup_two_stage/synthetic_scene_pickles \
  metric_cache_path=/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/exp_root/metric_cache_warmup_gpu_watch_20260413_075952 \
  worker=sequential
```

metric cache 目录：

- [metric_cache_warmup_gpu_watch_20260413_075952](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/exp_root/metric_cache_warmup_gpu_watch_20260413_075952)

对应的 cached token 是：

- `19e90f2757b25f38`

文件位置：

- [/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/exp_root/metric_cache_warmup_gpu_watch_20260413_075952/2021.09.16.19.27.01_veh-45_01749_03230/unknown/19e90f2757b25f38/metric_cache.pkl](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/exp_root/metric_cache_warmup_gpu_watch_20260413_075952/2021.09.16.19.27.01_veh-45_01749_03230/unknown/19e90f2757b25f38/metric_cache.pkl)

### 7.2 pdm score

随后执行 one-scene eval：

```bash
/data/zxz/condaenv/curious_vla/navsim/bin/python \
  /mnt/42_store/zxz/HUAWEI/VLA/curious_vla/navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py \
  train_test_split=warmup_two_stage \
  "train_test_split.scene_filter.log_names=['2021.09.16.19.27.01_veh-45_01749_03230']" \
  train_test_split.scene_filter.max_scenes=1 \
  train_test_split.scene_filter.include_synthetic_scenes=false \
  experiment_name=warmup_gpu_watch_20260413_075952 \
  agent=navsim_qwen_norm_cot_baseline_agent \
  "agent.config.model_name_or_path=/data/zxz/condaenv/curious_vla/models/Curious-VLA" \
  "+agent.config.api_base_url=http://127.0.0.1:8192/v1" \
  "+agent.config.max_tokens=512" \
  +agent.config.temperature=0.0 \
  navsim_log_path=/data/zxz/HUAWEI/VLA/navsim_data/navsim_logs/test \
  original_sensor_path=/data/zxz/HUAWEI/VLA/navsim_data/downloads/openscene-v1.1/sensor_blobs/test \
  synthetic_sensor_path=/data/zxz/HUAWEI/VLA/navsim_data/warmup_two_stage/sensor_blobs \
  synthetic_scenes_path=/data/zxz/HUAWEI/VLA/navsim_data/warmup_two_stage/synthetic_scene_pickles \
  metric_cache_path=/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/exp_root/metric_cache_warmup_gpu_watch_20260413_075952 \
  worker=sequential
```

## 8. 运行结果

这轮最终成功完成：

- `Number of successful scenarios: 1`
- `Number of failed scenarios: 0`

主结果：

- `PDMS (v1) = 0.8491359874973431`
- `EPDMS (v2) = 0.8706879892834369`
- `ego_progress = 0.6379263699936236`

其余这轮都是：

- `no_at_fault_collisions = 1.0`
- `drivable_area_compliance = 1.0`
- `driving_direction_compliance = 1.0`
- `traffic_light_compliance = 1.0`
- `time_to_collision_within_bound = 1.0`
- `lane_keeping = 1.0`
- `history_comfort = 1.0`

结果文件：

- [2026.04.13.08.05.43.csv](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/exp_root/warmup_gpu_watch_20260413_075952/2026.04.13.08.00.03/2026.04.13.08.05.43.csv)

详细日志：

- [run_pdm_score_one_stage.log](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/exp_root/warmup_gpu_watch_20260413_075952/2026.04.13.08.00.03/run_pdm_score_one_stage.log)
- [detailed_logs.jsonl](/mnt/42_store/zxz/HUAWEI/VLA/curious_vla/exp_root/warmup_gpu_watch_20260413_075952/2026.04.13.08.00.03/detailed_logs.jsonl)
- [warmup_gpu_watch_20260413_075936.log](/data/zxz/HUAWEI/VLA/navsim_data/logs/warmup_gpu_watch_20260413_075936.log)

## 9. 为什么我认为这次不是 fallback 假分

这次和之前那轮不一样，判断依据比较直接：

1. `run_pdm_score_one_stage.log` 里明确有：
   - `GET http://127.0.0.1:8192/v1/models 200 OK`
   - `POST http://127.0.0.1:8192/v1/chat/completions 200 OK`
2. 日志里没有：
   - `VLA client failed`
   - `Falling back`
   - `Agent failed`
3. `detailed_logs.jsonl` 里确实保存了模型返回的原始文本和解析后的轨迹

所以这轮结果应视为：

- 模型真正生成并被 parser 成功解析后得到的结果

而不是 constant velocity fallback 伪造出来的结果。

## 10. 这次模型的原始输出摘要

`detailed_logs.jsonl` 里保存的模型输出大意是：

- `critical_objects` 全部为 `no`
- `meta_behaviour.speed = decelerate`
- `meta_behaviour.command = straight`
- 输出了 8 个 `(x, y, yaw)` 未来点

解析并反归一化后的轨迹起点到终点大致是：

- 起点约 `(3.03, -0.01, -0.0065)`
- 终点约 `(15.89, -0.28, -0.0398)`

## 11. 时间开销

从 `run_pdm_score_one_stage.log` 看：

- `08:00:04` 开始真正 scoring
- `08:05:43` 收到成功的 `chat/completions`
- `08:05:43` 完成评测并落盘

所以这轮 one-scene end-to-end 评测耗时大约：

- `5 分 39 秒`

这不是单纯模型 forward 时间，而是整条链路耗时，包括：

- evaluator
- agent
- 图像处理
- HTTP 请求
- 服务端生成
- parser
- scoring

## 12. 给 NPU 侧复现时的建议

如果你想在 NPU 上尽量对齐这次 GPU 样例，建议优先保证下面几点一致：

1. 仍然使用同一个 log 约束：
   - `2021.09.16.19.27.01_veh-45_01749_03230`
2. 仍然限制：
   - `max_scenes=1`
   - `include_synthetic_scenes=false`
3. 原始图像路径必须还是 `test` 的 original sensor 数据，不要误指到 `warmup_two_stage/sensor_blobs`
4. `max_tokens` 先保持 `512`
5. `temperature` 保持 `0.0`
6. 如果要和这次 GPU 结果严格比，不要允许 fallback

### 12.1 最重要的对齐点

NPU 侧真正要对齐的不是探针 token `51ad5207706e5602`，而是这轮被 warmup split 最终选出来并评分的场景：

- `log_name = 2021.09.16.19.27.01_veh-45_01749_03230`
- `evaluated token = 19e90f2757b25f38`

### 12.2 如果你只是想做“同 log 对照”

那也可以先用探针 token 所在的 log 验证原始图像可读：

- `51ad5207706e5602`

但要知道：

- 它不是这次 CSV 对应的评分 token

## 13. 这次测试的局限

这轮只能说明：

- 当前 GPU 链路在 `original scene` 上已经至少有 1 个样例能真实跑通

但它还不能说明：

- 整个 warmup 平均分就是这个数
- NPU 和 GPU 的差异已经被严格归因
- 模型在更多 original scene 上的稳定性已经确认

所以这份记录更适合作为：

- 一个“可复现实验样例”
- 一个“排查 original scene / fallback / 服务健康度”的基准样例

