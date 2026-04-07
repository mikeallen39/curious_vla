# vLLM-Ascend 验证记录

本文记录在当前 Ascend NPU 机器上，为 `Curious-VLA` 打通
`vllm-ascend` 路径时的安装方法、验证过程、限制条件和当前结论。

## 1. 目标

这轮工作的目标不是一次性完成完整 benchmark 集成，而是先回答下面几个问题：

- 当前机器上能不能安装 `vllm-ascend`
- 能不能在 NPU 上加载本地 `Curious-VLA` 模型
- 能不能启动 OpenAI 兼容 API
- 能不能在真实 Curious-VLA prompt 下返回基本合理的结果

## 2. 机器与环境

当前机器环境：

- 架构：`aarch64`
- OS：`EulerOS 2.0 (SP10)`
- NPU：`Ascend 910B3`
- CANN：`8.1.RC1`

本次使用的独立环境：

- `/cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend`

模型目录：

- `/cache/ma-user/curious_vla_assets/models/Curious-VLA`

说明：

- 原来的 NPU latency 环境没有被改动
- 所有 `vllm-ascend` 相关工作都放在单独 env 中

## 3. 为什么标准安装不行

直接执行：

```bash
pip install vllm==0.9.1
```

在当前机器上不能直接成功，主要有两个原因。

### 3.1 没有 `aarch64` 现成 wheel

`vllm 0.9.1` 在 PyPI 上没有提供当前机器可直接用的 `aarch64` wheel，
只能拿到：

- `vllm-0.9.1.tar.gz`
- `vllm-0.9.1-cp38-abi3-manylinux1_x86_64.whl`

所以当前机器只能：

- 从源码装
- 或者从本地源码树安装

### 3.2 默认源码编译会走到 CPU custom ops

默认源码编译时，会尝试编译 upstream `vllm` 的 CPU custom ops，
而当前系统编译器会在这一阶段报错：

- 系统 `g++`：`7.3.0`
- 报错：
  `invalid feature modifier in '-march=armv8.2-a+dotprod+fp16'`

## 4. 当前可用的安装策略

最终可用的安装方案是：

1. 安装 `vllm-ascend==0.9.1`
2. 再安装 upstream `vllm`，但把 `VLLM_TARGET_DEVICE` 设成 `empty`

核心思路：

- `VLLM_TARGET_DEVICE=empty` 会跳过 upstream custom-op 编译
- `vllm-ascend` 仍然会提供 Ascend 平台所需的插件和补丁
- 这样就绕开了 `aarch64` 上的 CPU custom-op 编译问题

### 4.1 最终基线版本

当前 env 中验证可工作的核心版本：

- `vllm`: `0.9.1+empty`
- `vllm-ascend`: `0.9.1`
- `torch`: `2.5.1`
- `torch_npu`: `2.5.1.post1`
- `transformers`: `4.52.4`
- `numpy`: `1.26.4`

### 4.2 实际安装流程

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate /cache/ma-user/curious_vla_assets/envs/curious-vla-vllm-ascend

pip install vllm-ascend==0.9.1
```

然后从源码安装 upstream `vllm`：

```bash
pip download --no-deps vllm==0.9.1
tar -xf vllm-0.9.1.tar.gz
export VLLM_TARGET_DEVICE=empty
pip install --no-deps ./vllm-0.9.1
```

之后还补装了一批 CLI 和服务所需依赖，例如：

- `fastapi[standard]`
- `openai`
- `aiohttp`
- `tiktoken`
- `msgspec`
- `sentencepiece`
- `protobuf`
- `opencv-python-headless`

## 5. 设备映射的关键细节

这台机器虽然物理卡在 `npu-smi` 里显示为 `NPU 2`，
但当前容器对进程暴露出来的逻辑卡号只有：

- `0`

所以实际结论是：

- `ASCEND_RT_VISIBLE_DEVICES=0` 可用
- 不设置也可用
- `ASCEND_RT_VISIBLE_DEVICES=2` 不可用

错误现象包括：

- `aclrtGetDeviceCount` 失败
- visible device 配置非法

## 6. 烟雾测试结论

当前 env 下，这些动作已经成功：

```bash
python -c "import vllm; import vllm_ascend"
vllm --help
```

说明：

- `vllm` 本体可以 import
- Ascend 平台插件可以被发现
- CLI 可以启动

之后，用下面的命令成功拉起过基础服务：

```bash
vllm serve /cache/ma-user/curious_vla_assets/models/Curious-VLA \
  --host 127.0.0.1 \
  --port 18000 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

并且：

- `/v1/models` 可以返回
- `/v1/chat/completions` 可以返回结果

这说明系统层面的路径已经打通：

- 进程能起
- 模型能加载
- scheduler 能工作
- token 生成链路能工作

## 7. 为什么还不能直接说“可用”

虽然服务能启动，但烟雾测试只能说明：

- 系统路径打通了

不能说明：

- 输出一定合理
- 服务一定适合拿来测 benchmark

主要原因：

- `Curious-VLA` 是视觉语言规划模型，不是纯文本 chat 模型
- 纯文本 smoke test 不能代表真实规划 prompt
- 之前没有把数据集场景接进这条服务路径
- 还没有任何 planning 质量门槛

## 8. 语义验证是怎么补上的

后来补了一套轻量级语义 gate，入口是：

- `local/run_vllm_semantic_validation.py`
- `local/run_vllm_semantic_validation.sh`

这套 gate 不做完整 PDM，而是做三层检查：

1. text-only schema control
2. text-only planning-style control
3. 基于真实 warmup scene 和真实前视图图像的 VL planning 校验

主要验证项包括：

- 严格 JSON 合同
- `critical_objects` 是否完整
- `meta_behaviour` 值域是否合法
- 预测的 `command` 是否和 scene intent 对齐
- 轨迹能否解析
- 反归一化后的轨迹是否合理
- 和 warmup 仅有的 1 个 future 点之间的一步误差是否过大

## 9. 为什么后来要加图像缩放

原始前视图是：

- `1920x1080`

这会导致真实 planning prompt 太长。

对同一条真实 scene 的 processor 侧估算结果：

- `1920x1080`：约 `3603`
- `1280x704`：约 `2062`
- `960x540`：约 `1558`
- `640x360`：约 `1211`

这意味着：

- `2048` context 可以支撑 `960x540`
- 但对 `1280x704` 或原图就不够宽裕
- 把 `max-model-len` 提得太高又容易在 profile 阶段触发 NPU OOM

因此最后形成了两个可用区间：

- `960x540 + max-model-len=2048`
- `1280x704 + max-model-len=2560 + max-num-batched-tokens=2560 + max-num-seqs=1`

## 10. 当前实际验证结果

### 10.1 `960x540` 小批量 VL 语义 gate

报告：

- `/cache/ma-user/curious_vla_assets/logs/vllm_semantic_validation_20260407_batch6.json`

结果摘要：

- scene 数：`6`
- request 成功：`6/6`
- overall valid：`6/6`
- mean latency：`12.979475s`

### 10.2 `1280x704` 语义 gate

报告：

- `/cache/ma-user/curious_vla_assets/logs/vllm_semantic_validation_1280x704_20260407.json`

结果摘要：

- scene 数：`2`
- request 成功：`2/2`
- overall valid：`2/2`
- mean latency：`15.245859s`

这一步很关键，因为它证明了：

- 当前机器上 `1280x704` 不是理论可行，而是已经实际跑通过

## 11. 当前结论

到目前为止，可以下的结论是：

- 当前机器上可以拉起 `vllm-ascend`
- 可以用它服务本地 `Curious-VLA`
- 可以在真实 planning prompt 下完成轻量级语义 gate
- 可以在 gate 通过后继续做服务化 latency benchmark

但还不能下这些结论：

- 已经通过完整 PDM / EPDMS 质量验证
- 原始 `1920x1080` 服务形态已经稳定
- 这条路径已经能完全替代本地 `transformers` benchmark

## 12. 最推荐的使用顺序

当前最稳妥的顺序是：

1. 按已验证配置启动 `vllm-ascend`
2. 先跑语义校验
3. 只有校验通过时，再跑 latency benchmark

这样得到的 latency 会比“服务能返回点东西”更有意义。

如果你只想看更完整的 `1280x704` 总结，请继续看：

- `vllm_ascend_1280x704_total_report.md`
