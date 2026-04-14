# Curious-VLA 模型架构与 Forward 全流程分析

## 1. 结论先行

当前仓库里用于 NAVSIM 评测的 `Curious-VLA`，本质上不是一个“直接回归轨迹的专用网络头”，而是一个：

- 以 `Qwen2_5_VLForConditionalGeneration` 为运行时主体的视觉语言生成模型
- 输入是图像 + 文本 prompt
- 输出仍然是自然语言 / 结构化文本
- 轨迹不是由单独的 regression head 直接输出，而是由模型 `generate()` 出文本后，再通过正则 parser 解析成 `8 x 3` 的归一化轨迹点

也就是说，当前推理栈的核心形态是：

1. 场景数据整理成 `front image + ego history + high-level command`
2. 组织成 VLM 对话 prompt
3. 调 OpenAI 兼容接口或本地 HuggingFace `generate()`
4. 从文本里解析 `[(x, y, yaw), ...]`
5. 用统计量反归一化成真实轨迹
6. 返回给 NAVSIM evaluator 计算 PDM / EPDMS

因此，延迟和稳定性都不只取决于“模型前向”，还明显受以下环节影响：

- 图像落盘和重新读取
- base64 编码
- HTTP 调用
- 文本生成长度
- 文本解析成功率
- fallback 策略

## 2. 当前实际运行的模型是什么

本地模型目录是：

- `/data/zxz/condaenv/curious_vla/models/Curious-VLA`

从 `config.json` 看，运行时架构明确是：

- `architectures = ["Qwen2_5_VLForConditionalGeneration"]`
- `model_type = "qwen2_5_vl"`

从模型目录 `README.md` 看，这个模型的 Hugging Face 元信息写的是：

- `base_model: Qwen/Qwen2.5-3B-Instruct`

这里要注意区分两层含义：

- `README` 里的 `base_model` 更像训练来源/元信息
- 真正运行时加载的类，以 `config.json` 为准，是 `Qwen2.5-VL` 条件生成模型

所以，应该把它理解为：

- 语言骨干接近 `Qwen2.5 3B` 这一量级
- 但实际部署体是带视觉分支的 `Qwen2.5-VL`

模型文件索引里记录的总权重大小约为：

- `8.13 GB`（`model.safetensors.index.json` 的 `metadata.total_size`）

## 3. 模型结构拆解

### 3.1 文本主干

从 `config.json` 读取到的关键文本结构参数：

- `hidden_size = 2048`
- `intermediate_size = 11008`
- `num_hidden_layers = 36`
- `num_attention_heads = 16`
- `num_key_value_heads = 2`
- `max_position_embeddings = 128000`
- `vocab_size = 151669`
- `hidden_act = silu`
- `rms_norm_eps = 1e-6`

这说明文本部分是一个标准 decoder-only causal LM 主干，按 token 自回归生成输出。

### 3.2 视觉主干

`vision_config` 的关键参数：

- `depth = 32`
- `hidden_size = 1280`
- `intermediate_size = 3420`
- `num_heads = 16`
- `patch_size = 14`
- `spatial_merge_size = 2`
- `out_hidden_size = 2048`
- `fullatt_block_indexes = [7, 15, 23, 31]`
- `window_size = 112`

可以把它理解为：

- 输入图像先经过视觉 patch / merge 编码
- 视觉塔输出的特征再投影到文本侧 hidden size `2048`
- 随后把视觉 embedding 插入文本 token 序列中的 image token 位置
- 最后统一进入语言模型做联合推理

### 3.3 输出头

在训练侧 monkey patch 代码里，`qwen2_vl_model_forward()` 很直接：

- `hidden_states = self.model(...)`
- `logits = self.lm_head(hidden_states)`

说明当前输出头仍然是标准语言模型 `lm_head`，没有看到仓库里单独定义的轨迹回归 head。

这也是为什么轨迹最终是“从文本里解析出来”，而不是直接从张量头回归出来。

## 4. 当前 GPU 评测链路的整体结构

你现在在 GPU 上跑 NAVSIM eval，实际走的是下面这条链：

1. `run_pdm_score_one_stage.py`
2. Hydra instantiate `NavsimCoTQwenAgent`
3. `agent.initialize()`
4. `CuriousVLAClient(...)`
5. OpenAI-compatible 接口 `chat.completions.create(...)`
6. 后端服务由 `llamafactory-cli api` 启动
7. 服务内部加载 `Curious-VLA` 模型并执行生成
8. 返回文本
9. `CuriousVLAClient` 正则解析轨迹
10. agent 反归一化后返回 `Trajectory`
11. NAVSIM evaluator 用该轨迹打分

其中本地启动服务的脚本是：

- `local/start_lf_server.sh`
- `local/watch_and_run_warmup_gpu_smoke.sh`

服务启动参数的关键点是：

- `--model_name_or_path $CURIOUS_VLA_MODEL_DIR`
- `--template qwen2_vl`
- `--infer_backend huggingface`
- `--image_max_pixels 262144`
- `--trust_remote_code true`

这说明：

- 服务端不是 vLLM 路径，而是 HuggingFace backend
- 对话模板按 `qwen2_vl` 处理
- 图像大小有服务侧上限

## 5. 从 evaluator 到 agent 的 forward 调用链

### 5.1 evaluator 入口

`navsim_eval/navsim/planning/script/run_pdm_score_one_stage.py` 的关键过程是：

1. 构造 `SceneLoader`
2. 构造 `MetricCacheLoader`
3. 对每个 token：
   - `agent_input = scene_loader.get_agent_input_from_token(token)`
   - `scene = scene_loader.get_scene_from_token(token)`
   - `trajectory = agent.compute_trajectory(agent_input, scene)`
4. 再把 `trajectory` 交给 `pdm_score(...)`

这里真正的“模型 forward”入口，不是在 evaluator 里，而是在：

- `NavsimCoTQwenAgent.compute_trajectory(...)`

### 5.2 agent 初始化

`NavsimCoTQwenAgent.initialize()` 会创建：

- `CuriousVLAClient`

其中配置来源是：

- `CuriousVlaConfig`

关键字段有：

- `model_name_or_path`
- `api_base_url`
- `api_key`
- `max_tokens`
- `temperature`
- `allow_missing_front_camera_fallback`
- `log_path`

## 6. Agent 侧的完整 forward 流程

下面按 `NavsimCoTQwenAgent.compute_trajectory()` 的真实执行顺序展开。

### 6.1 读取当前 token 对应的场景输入

agent 收到的是：

- `AgentInput`
- `Scene`

其中会用到：

- `scene.scene_metadata.initial_token`
- `scene.scene_metadata.log_name`
- `agent_input.cameras`
- `agent_input.ego_statuses`

### 6.2 检查 front camera 是否存在

当前代码会先拿：

- `current_cams = agent_input.cameras[-1]`
- `front_camera = current_cams.cam_f0`
- `image_np = front_camera.image`

如果 `front_camera.image` 缺失：

- 现在默认会直接 `RuntimeError`
- 只有 `allow_missing_front_camera_fallback=True` 时才会退回 constant velocity

这部分是你之前要求改成“original 数据不允许静默 fallback”的地方。

### 6.3 决定需要哪些相机

`get_sensor_config()` 决定 SceneLoader 会提前取哪些传感器：

- 默认 `cam_type='single'`，只请求 `cam_f0`
- `multi_view` 会请求 6 个相机
- `cont` 会请求连续多个历史 front 帧

但当前本地实际评测默认还是：

- `single`

所以默认只吃当前时刻 front view。

### 6.4 把 numpy 图像写成临时 JPG

agent 并不是直接把内存里的图像张量传给服务端，而是先：

- 把 `cam_data.image` 从 numpy array 转成 `PIL.Image`
- 保存到 `self._temp_dir/step_xxx_cam_f0.jpg`

这一步的后果是：

- 每个 token 都有一次图像落盘
- 之后 client 又会重新打开这个文件进行 base64 编码

这是当前链路里的额外 CPU / IO 开销点。

### 6.5 构造 prompt

`_build_prompt_messages()` 会把以下信息拼进文本：

- 当前高层导航意图 `driving_command`
- 过去 `1.5s` 的 ego history
- 当前图像说明
- 任务说明

默认单图模式下，prompt 不是只要求轨迹，而是四阶段 COT 风格：

1. `critical_objects`
2. `explanation`
3. `meta_behaviour`
4. `future_trajectory`

最终要求模型输出一个严格 JSON 风格文本，其中重点字段是：

- `"future_trajectory": [PT, ...]`

每个点格式是：

- `(x, y, heading)`

并且要求总共输出：

- `8` 个 future 点
- 采样频率 `2 Hz`
- 预测时长 `4s`

### 6.6 调 client.forward()

agent 之后调用：

- `self._client.forward(messages, use_yaw_parser=True)`

这里的 `messages` 仍然是仓库自定义格式：

- `{"images": [...], "messages": [...]}`

还不是 OpenAI 原生格式。

## 7. Client 侧的完整 forward 流程

### 7.1 转成 OpenAI 多模态消息格式

`CuriousVLAClient.forward()` 首先调用：

- `convert_to_openai_format(messages, dataset_dir=None)`

这个转换器做两件事：

1. 读取 `images` 列表中的图片文件
2. 编码为 `data:image/jpeg;base64,...`

然后把它们注入第一条带 `<image>` 占位符的 user message，形成：

- `{"type": "image_url", "image_url": {"url": data_uri}}`
- `{"type": "text", "text": "..."}`

所以服务端看到的是标准 OpenAI 多模态 chat payload。

### 7.2 调 OpenAI-compatible 接口

之后 client 通过官方 `openai` Python SDK 调：

- `client.chat.completions.create(...)`

关键参数：

- `messages=llm_messages`
- `model=self.model_name_or_path`
- `max_tokens=self.max_tokens`
- `temperature=request_temperature`

也就是说，这里虽然叫 `model=...`，但并不是远程 SaaS 模型名，而是发给本地 LlamaFactory 服务的模型标识。

### 7.3 服务失败重试

`CuriousVLAClient.forward()` 有 3 次重试：

- 随机选可用 server
- 若异常则指数退避
- 最后仍失败则抛异常

但再往上层 `agent.compute_trajectory()` 里，这个异常目前仍会被捕获，并 fallback 到 constant velocity。

这意味着：

- 数据缺失现在可以硬失败
- 但服务错误/解析错误目前仍可能被 fallback 掩盖

这一点对你做严格 benchmark 时是重要风险点。

### 7.4 文本解析成轨迹

当前评测默认用：

- `parse_trajectory_string_with_yaw()`

它通过正则匹配：

- `[(x, y, yaw), ...]`
- 或 `(x, y, yaw)` 风格

只要能匹配出至少 `8` 个点，就取前 `8` 个。

这里再次说明：

- 模型不是直接输出轨迹 tensor
- 而是输出文本
- parser 再把文本转成 `torch.Tensor(8, 3)`

## 8. 服务端内部实际在做什么

虽然当前 GPU eval 走的是 LlamaFactory 服务，但从本仓库的 NPU 直连脚本可以比较清楚地还原服务端实际语义流程。

`local/run_planning_latency_benchmark_npu.py` 里定义的 `LocalCuriousVLAClientNPU`，本质上就是把服务端逻辑在本地 Python 里直接展开了。

其核心步骤如下。

### 8.1 AutoProcessor 组装文本和图像

先加载：

- `AutoProcessor.from_pretrained(...)`

然后：

1. 读取图片
2. 转成 RGB
3. 默认 resize 到 `1280 x 704`
4. 构造 chat messages
5. `processor.apply_chat_template(...)`
6. `processor(..., images=[processed_image], return_tensors="pt")`

processor 输出的关键张量通常包括：

- `input_ids`
- `attention_mask`
- `pixel_values`
- `image_grid_thw`

这一步完成后，文本 token 和视觉输入都已经准备好。

### 8.2 generate()

随后调用：

- `self.model.generate(**inputs, max_new_tokens=..., do_sample=...)`

这里模型类就是：

- `Qwen2_5_VLForConditionalGeneration`

### 8.3 decode

生成完以后，会把新增 token 解码成字符串：

- `processor.batch_decode(new_tokens, skip_special_tokens=True)[0]`

### 8.4 parse

然后还是用同样的 parser：

- `parse_trajectory_string_with_yaw(output_text)`

因此，无论是：

- GPU 上的服务化链路
- 还是 NPU 上的直连 `generate()` 链路

它们在“模型语义层”上都是同一件事：

- 多模态输入
- 自回归文本生成
- 文本解析轨迹

## 9. Qwen2.5-VL 模型内部 forward 的关键步骤

这一层在仓库里最清楚的实现，不是 eval agent，而是训练侧的 monkey patch：

- `EasyR1/verl/models/transformers/qwen2_vl.py`

虽然它主要服务于训练/并行 patch，但非常适合拿来理解模型内部的真正张量流。

### 9.1 文本 token embedding

首先：

- `inputs_embeds = model.get_input_embeddings()(input_ids)`

也就是先把文本 token 变成 embedding。

### 9.2 视觉编码

如果有图像：

- `image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)`

这里视觉塔把图像 patch 序列编码成视觉特征。

### 9.3 视觉特征回填到文本序列

之后用 image token 位置的 mask，把视觉 embedding scatter 回文本序列：

- 找出 `input_ids == image_token_id`
- 把 `image_embeds` 填进 `inputs_embeds` 对应位置

所以最终送给语言模型的，不是纯文本 embedding，而是：

- 文本 embedding
- 混合了视觉特征后的多模态 embedding 序列

### 9.4 位置编码

Qwen2-VL 不是简单的一维文本位置编码，而是会为视觉部分构造 3D 位置索引：

- 时间维 `t`
- 高度维 `h`
- 宽度维 `w`

训练侧 patch 里的 `get_rope_index()` 就是在做这件事。

这说明视觉 token 在模型内部不是被当成普通文本 token 粗暴拼接，而是带有显式的多维位置结构。

### 9.5 语言模型主干

随后调用：

- `self.language_model(input_ids=None, **kwargs)`

也就是标准 decoder-only transformer 主干。

### 9.6 lm_head 输出 logits

最后：

- `logits = self.lm_head(hidden_states)`

再交给 `generate()` 做自回归采样/贪心生成。

所以从张量角度看，这个模型的 forward 可以概括为：

1. 文本 token embedding
2. 图像编码成视觉 embedding
3. 视觉 embedding 替换 image token 位置
4. 多模态序列送入 decoder-only transformer
5. `lm_head` 输出下一个 token logits
6. `generate()` 反复迭代直到结束

## 10. 轨迹是如何从“文本”变成“可评分轨迹”的

### 10.1 解析出的还是归一化轨迹

模型 parser 输出的是：

- `torch.Tensor(8, 3)`

但这仍然是训练统计量归一化空间下的轨迹。

### 10.2 反归一化

agent 里会读取：

- `stats/trajectory_stats_train.json`

然后执行：

- `denormalize = poses * std + mean`

最终才得到真实尺度上的：

- `(x, y, yaw)` future trajectory

### 10.3 封装成 NAVSIM 的 `Trajectory`

最后封装为：

- `Trajectory(parsed_trajectory, trajectory_sampling)`

默认采样配置是：

- `time_horizon = 4.0`
- `interval_length = 0.5`

也就是：

- 4 秒
- 2 Hz
- 8 个点

## 11. 当前链路里的主要延迟热点

如果从性能角度看，当前 forward 不是一个“只有一次 model.forward()”那么简单，而是至少包含下面这些阶段：

1. `SceneLoader` 取数据
2. numpy 图像转 JPG 落盘
3. client 重新读图并 base64 编码
4. HTTP 请求
5. processor 做图文打包
6. 视觉塔前向
7. 语言模型自回归生成
8. 文本解码
9. 正则解析
10. 轨迹反归一化

对单 token latency 来说，通常最重的仍然是：

- 视觉编码
- 自回归生成

但当前代码里额外的工程开销也不小：

- 图像磁盘往返
- base64 编码
- HTTP 序列化

如果后面要严肃优化 latency，这几个点都值得单独拆开测。

## 12. 当前实现中的几个关键 caveat

### 12.1 这是“文本生成轨迹”，不是“直接回归轨迹”

这意味着：

- 输出受 prompt 写法强影响
- 输出格式错误会直接导致 parse 失败
- 解析成功率和服务稳定性会显著影响最终 benchmark

### 12.2 仍然存在 client 失败 fallback

当前 `compute_trajectory()` 里：

- 缺图像现在可以硬失败
- 但 `CuriousVLAClient.forward()` 的异常仍会被 catch 后 fallback 到 constant velocity

所以如果要做严格 GPU 对比，建议后续把这条 fallback 也禁掉。

### 12.3 `multi_view` 路径目前有格式风险

`convert_to_openai_format()` 只有在 user message 里检测到 `<image>` 时才会注入图片。

但 `multi_view` 的 prompt 文本当前没有显式 `<image>` 占位符。

这意味着：

- `single` 模式现在是通的
- `multi_view` 路径按当前实现看，存在图片未被正确注入 OpenAI message 的风险

这不是本文主线，但如果后面要切多视角评测，这一处需要单独检查。

## 13. 一句话总结

当前 Curious-VLA 在本仓库里的推理本质是：

- `Qwen2.5-VL` 多模态生成模型
- 用 front image + ego history + command 构造驾驶 prompt
- 通过 `generate()` 产出文本化轨迹
- parser 把文本提取成 `8 x 3` 归一化轨迹
- 再反归一化后进入 NAVSIM 评测

因此它的“forward 全流程”必须看成一条端到端链路，而不是只看 transformer 主干里的那一次矩阵乘法。

