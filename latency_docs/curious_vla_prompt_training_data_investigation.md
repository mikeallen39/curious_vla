# Curious-VLA 训练 Prompt 与公开模型行为排查记录

本文整理 2026-04-09 针对当前公开 `Curious-VLA` 模型的一轮深入排查，重点回答以下问题：

- 当前本地模型是否下载错误
- 当前 benchmark 使用的 prompt 是否偏离训练分布
- 为什么 `critical_objects` 在 `warmup_two_stage` 中几乎总是全 `no`
- `trajonly` / `cpr` 这类更“像源码里的训练范式”的 prompt，是否能显著改善输出

## 1. 核心结论

本轮排查后的结论如下：

1. 本地模型没有下载错。
   - 本地目录 `/cache/ma-user/curious_vla_assets/models/Curious-VLA` 与 Hugging Face 仓库 `MashiroLn/Curious-VLA` 完全对齐。
   - 关键文件哈希、文件列表、参数规模都一致。

2. 当前公开训练数据里的真实 `problem`，并不是 `trajonly` 或 `cpr` 风格。
   - 真实训练数据使用的是和当前 agent 非常接近的多任务 prompt：
     - `critical_objects`
     - `explanation`
     - `meta_behaviour`
     - `future_trajectory`
   - 同时还显式要求：
     - `<thinking>...</thinking>`
     - `<answer>...</answer>`

3. 当前 agent prompt 与真实训练 prompt 确实有轻微偏差，但不是主要问题。
   - 把 warmup 场景改成“和真实训练数据一致”的 prompt 后：
     - 轨迹误差只轻微改善
     - `critical_objects` 仍然稳定全 `no`

4. `critical_objects` 持续塌缩，更像是训练目标本身导致，而不是单纯 prompt 没对齐。
   - 当前公开 RL 数据里 `answer['gt']` 是空数组，仅保留 `token`
   - reward 只真正关心轨迹，对显式语义字段没有强约束
   - 这会允许模型把 `critical_objects` 学成低成本占位策略

5. `trajonly` / `cpr` prompt 并没有表现出更强优势。
   - `trajonly` 只在少数样本上略微降低首步误差，但格式稳定性明显更差
   - `cpr` 基本无法稳定输出可解析轨迹

## 2. 模型是否下载错误

### 2.1 本地模型目录

本地模型目录：

- `/cache/ma-user/curious_vla_assets/models/Curious-VLA`

关键文件包括：

- `config.json`
- `chat_template.jinja`
- `generation_config.json`
- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`
- `model.safetensors.index.json`
- `preprocessor_config.json`
- `tokenizer_config.json`
- `README.md`

### 2.2 与 Hugging Face 镜像核对结果

通过 `hf-mirror` 查询：

- 模型仓库：`MashiroLn/Curious-VLA`
- repo sha：`8ed0dcec9a228cfec16cb1aac2fbec277cd29373`
- `siblings` 文件列表与本地一致
- `safetensors.parameters.BF16 = 4064694272`

本地 `model.safetensors.index.json` 中的总参数量同样为：

- `4064694272`

另外对以下文件做了本地与远端逐个 SHA256 比对，全部一致：

- `README.md`
- `config.json`
- `chat_template.jinja`
- `generation_config.json`
- `model.safetensors.index.json`
- `preprocessor_config.json`
- `special_tokens_map.json`
- `tokenizer_config.json`

### 2.3 关于“是不是下成了 adapter / 残缺权重”

排查结果：

- 没有 `adapter_config.json`
- 没有 `adapter_model.safetensors`
- 权重是完整两片式 safetensors

因此可以排除：

- 下错仓库
- 只下到 adapter
- 本地模型文件不完整

### 2.4 仍需保留的谨慎结论

虽然本地没有下载错，但仍不能完全证明：

- `MashiroLn/Curious-VLA` 这份公开 release
- 就等于论文 benchmark 时使用的最终最佳 checkpoint + 最终最佳推理 recipe

因为 `config.json` 里还能看到一个明显像训练中间导出路径的痕迹：

- `text_config._name_or_path = /mnt/data/ccy/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b_navsim_adas_3x_1k/global_step_130/actor/huggingface`

它更像某个 RL actor checkpoint，而不是一个明确命名为 paper-best 的最终打包结果。

## 3. 真实训练数据到底长什么样

### 3.1 数据仓库

通过镜像确认，训练数据仓库为：

- `MashiroLn/Curious-VLA` dataset

其中公开文件只有：

- `EasyR1/data/QA_navtrain_poutine_style_full/data/train.parquet`
- `EasyR1/data/QA_navtrain_poutine_style_full/data/test.parquet`

### 3.2 数据规模

实测：

- `test.parquet`: `100` 条
- `train.parquet`: `103288` 条

字段都只有三列：

- `images`
- `problem`
- `answer`

### 3.3 一个非常重要的发现

在全量 `train.parquet` 中：

- `103288 / 103288` 的 `answer['gt']` 都是空数组

也就是：

- `answer = {"gt": [], "token": ...}`

这与 `docs/train_grpo.md` 里“`gt is useless in RLVR`”的描述是一致的。

这说明当前公开出来的这套 parquet，本质上更像：

- RL 阶段使用的 prompt + token 数据

而不是：

- 带有显式标准答案文本监督的 SFT 数据

## 4. 真实训练 Prompt Skeleton

对前 `5000` 条训练样本做归一化后，只有 `1` 个统一的 prompt skeleton。

其结构核心如下：

1. `Task 1: Critical Objects and Conditions Detection`
2. `Task 2: Natural Language Explanation`
   - 明确要求 `thinking output should be wrapped in <thinking>...</thinking>`
3. `Task 3: Meta-Behaviour Selection`
4. `Task 4: Future Trajectory Prediction`
   - 明确要求 `answer output should be wrapped in <answer>...</answer>`
   - `future_trajectory` 字段的目标格式写成：
     - `"<answer>[PT, ...]</answer>"`

这意味着：

- 当前公开训练数据并不是 trajectory-only
- 而是一个显式多任务 JSON 规划 prompt

## 5. 当前 Agent Prompt 与训练 Prompt 的差异

对真实训练数据里的 prompt skeleton，与当前 agent 构造的 prompt 做了 diff。

主要差异只有三处：

1. 当前 agent 去掉了 `Task 2` 中的 `<thinking>...</thinking>` 约束
2. 当前 agent 把 `Task 4` 的输出提示从：
   - `<answer>...</answer>`
   改成了更模糊的：
   - `...`
3. 当前 agent 把 `future_trajectory` 的目标格式从：
   - `"<answer>[PT, ...]</answer>"`
   改成了：
   - `[PT, ...]`

换句话说：

- 当前 agent prompt 和真实训练分布并不是完全一致
- 但差异并不大，仍属于同一 prompt family

## 6. Prompt 对照实验

### 6.1 实验对象

选取：

- `warmup_two_stage/openscene_meta_datas` 中前若干真实场景

使用当前仍能工作的公开模型服务路径：

- `llamafactory-cli api`
- OpenAI 兼容接口

### 6.2 第一组实验：`official_multitask` vs `trajonly_style` vs `cpr_style`

分别构造三类 prompt：

1. `official_multitask`
   - 基本等同当前 agent 使用的 prompt
2. `trajonly_style`
   - 使用 `EasyR1/examples/format_prompt/trajonly.jinja`
3. `cpr_style`
   - 使用 `EasyR1/examples/format_prompt/cpr.jinja`

在 `10` 个真实 warmup 场景上的结果：

- `official_multitask`
  - `parse_ok_rate = 1.0`
  - `json_ok_rate = 1.0`
  - `mean_first_step_xy_error ≈ 2.44 m`
  - `mean_critical_yes_count = 0`

- `trajonly_style`
  - `parse_ok_rate = 0.5`
  - `mean_first_step_xy_error ≈ 2.38 m`

- `cpr_style`
  - `parse_ok_rate = 0.1`
  - 唯一可解析样本的首步误差约 `3.16 m`

结论：

- `trajonly` 并没有表现出明显更接近训练态
- `cpr` 反而极不稳定
- 当前公开模型最稳定适配的仍是 `official_multitask`

### 6.3 第二组实验：`with_system` vs `no_system`

为了更接近 EasyR1 数据加载逻辑，又测试了：

- 是否显式加 `system: You are an expert driver.`

结果：

- 去掉 `system` 后，结果只发生很小变化
- 无法解释当前主要问题

结论：

- `system` message 不是主因

### 6.4 第三组实验：`current_agent_prompt` vs `exact_train_prompt`

又构造了两组更直接的对照：

1. `current_agent_prompt`
   - 当前 agent 实际在用的 prompt
2. `exact_train_prompt`
   - 直接把真实 parquet 中的 prompt skeleton 抽出来
   - 只替换当前场景的：
     - `<INTENT>`
     - `<HISTORY>`

再分别测试 `with_system` / `no_system`。

在 `10` 个真实 warmup 场景上的结果：

- `current_agent_with_system`
  - `parse_ok_rate = 1.0`
  - `mean_err1 ≈ 2.4404 m`
  - `mean_yes_count = 0`

- `current_agent_no_system`
  - `parse_ok_rate = 1.0`
  - `mean_err1 ≈ 2.4377 m`
  - `mean_yes_count = 0`

- `exact_train_with_system`
  - `parse_ok_rate = 1.0`
  - `mean_err1 ≈ 2.4155 m`
  - `mean_yes_count = 0`

- `exact_train_no_system`
  - `parse_ok_rate = 1.0`
  - `mean_err1 ≈ 2.4076 m`
  - `mean_yes_count = 0`

结论：

- 换成和真实训练数据一致的 prompt 后，轨迹误差只有非常轻微的改善
- `critical_objects` 依旧稳定全 `no`
- 这说明 prompt 偏差确实存在，但不是主要矛盾

## 7. 为什么 `critical_objects` 仍然全 `no`

综合本轮排查，更合理的解释是：

1. 公开 RL 数据中没有显式答案监督。
   - `answer['gt']` 全为空

2. 公开 reward 只真正关心轨迹。
   - `EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py`
   - 最终实际使用的是 simulator 的 `scaled_pdms`
   - `format_score` 直接固定成 `1.0`

3. 当前 agent 虽然要求输出：
   - `critical_objects`
   - `explanation`
   - `meta_behaviour`
   - `future_trajectory`
   但真正参与后续评测的，仍然只有轨迹

因此模型完全可能学出一种低成本策略：

- `critical_objects` 统一填 `no`
- `explanation` 写模板化废话
- `meta_behaviour` 只给少数高频标签
- 主要把生成能力放在轨迹字段上

这正好和我们在正式 benchmark 日志中观察到的现象一致。

## 8. 当前最可靠的判断

截至目前，更可靠的判断是：

1. 不是本地模型下载错了。
2. 不是我们误以为训练 prompt 是 `trajonly/cpr`。
3. 当前 agent prompt 与训练 prompt 有轻微不一致，但不是主因。
4. 真正主因更像是：
   - 当前公开 release 中，显式语义字段没有被强监督
   - reward 也不要求它们语义正确
   - 因此 `critical_objects` 全 `no` 是训练上被允许的塌缩策略

## 9. 对当前问题的最终回答

如果把最初的问题简化成一句话：

> 为什么 `warmup_two_stage` 里 `critical_objects` 看起来明显不对？

当前最合理的回答是：

- 因为这不是一个简单的“prompt 写错”或“模型下载错”问题
- 而是公开模型本身在当前训练目标下，没有被充分逼迫去学会可靠的显式语义输出

## 10. 这是否说明公开 checkpoint 本身没有学好

当前更稳妥的结论是：

- 可以认为这份公开 checkpoint 在我们当前复现条件下，确实没有把目标能力学好
- 但还不能直接下结论说“论文里的最终模型也同样有问题”

这里需要区分三件不同的事。

### 10.1 已经基本排除的问题

截至目前，以下问题已经基本排除：

1. 模型没有下载错。
   - 本地模型目录与 Hugging Face 仓库 `MashiroLn/Curious-VLA` 对齐
   - 文件列表、关键配置文件哈希、参数规模都一致

2. 模型不是残缺权重。
   - 本地是完整 `safetensors` 权重
   - 不是只下载到了 adapter

3. 当前异常不主要是 prompt 偏差导致。
   - 即便切到更接近真实训练数据的 prompt family
   - `critical_objects` 仍然稳定全 `no`
   - 轨迹误差也只得到很小改善

因此，问题已经不能简单归因于：

- 下错模型
- 权重损坏
- prompt 没对齐

### 10.2 为什么说“这份公开 checkpoint 没有学好”

支持这个判断的证据主要有三条。

第一，显式语义字段长期塌缩。

- `critical_objects` 在真实 warmup 场景中几乎总是全 `no`
- `meta_behaviour` 也明显塌缩到少数高频标签
- 这说明模型没有把这些中间语义表示学成可靠输出

第二，问题不只在语义字段，轨迹本身也会明显出错。

一个代表性失败样例是：

- token: `6cca874fda7a3507a`

该场景前方存在车辆，同时有红灯语义线索。模型输出却是：

- `critical_objects`: 全 `no`
- `meta_behaviour.speed = decelerate`
- `meta_behaviour.command = straight`

对应反归一化后的 4 秒轨迹最终仍前进约 `20.94 m`，并不是接近停车的轨迹。

更重要的是，这个样例在 benchmark 中并不是“语义错了但轨迹还行”，而是直接被判成坏轨迹：

- `score = 0.0`
- `pdms_v1 = 0.0`
- `epdms_v2 = 0.0`
- `no_at_fault_collisions = 0.0`
- `time_to_collision_within_bound = 0.0`
- `lane_keeping = 0.0`

这说明当前公开 checkpoint 的问题不只是：

- 不会把物体写出来

而是更接近：

- 场景理解不够稳定
- 安全停车决策也没有学扎实

第三，公开训练信号本身就不足以强迫这些字段学好。

- 公开 parquet 中 `answer['gt']` 为空
- reward 最终主要使用 simulator 返回的 `scaled_pdms`
- `format_score` 在实现中直接固定为 `1.0`

这意味着模型在训练中更容易学到一种低成本策略：

- 语义字段尽量模板化
- 主要把生成能力压在轨迹字段上

### 10.3 为什么还不能直接等同于“论文模型有问题”

虽然我们可以说：

- 当前公开 release 在本地复现下没有学好

但还不能直接推出：

- 论文中报告结果对应的最终模型也一定同样失败

原因是：

1. 公开 release 可能不是论文最终最佳 checkpoint。
   - `config` 中仍保留了明显像训练中间导出路径的痕迹
   - 更像某个 RL actor 导出点，而不一定是 paper-best 打包版

2. 论文 benchmark 使用的推理 recipe 可能和公开仓库当前可运行路径不完全一致。
   - 包括 prompt 细节
   - 输出模板
   - 解析方式
   - multi-view / single-view 路径差异

3. 公开数据并不能完整覆盖论文中的全部训练阶段。
   - 当前公开 parquet 更像 RL 阶段数据
   - 尚未看到足以强监督语义字段的完整 SFT 数据

因此更准确的说法应该是：

- 当前公开 checkpoint 在我们现在拿到的公开代码、公开数据、公开推理路径下，表现出明显的“没学好”
- 但它是否等同于论文最终用于报分的完整系统，当前还没有证据能完全确认

## 11. 建议下一步

如果继续深挖，最有价值的下一步是：

1. 查未公开的 SFT 阶段痕迹
   - 目前公开 parquet 明显不是带标准答案文本的 SFT 数据
   - 需要确认论文效果是否依赖一个未公开或尚未发布的更强 SFT 阶段

2. 继续把失败样例做成 casebook
   - 图像
   - 当前 prompt
   - 原始输出
   - GT 首步
   - 预测首步
   - benchmark 子项分数

3. 如果目标是实用评测而不是复现论文范式，可以考虑单独增加一个更合理的 semantic gate
   - 只把 `critical_objects` 当作辅助诊断
   - 不再把它误认为模型具备稳定的显式场景理解能力
