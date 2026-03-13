# How to Deploy And Eval Curious-VLA?

Curious-VLA model is based on Qwen2.5-VL-3B, mainly focusing on the Training Paradigm.

The **Deployment** of Curious-VLA is the same as Qwen2.5-VL-3B.

## 1. Deploy (Server)

1. **Env1: lf**: Create LLaMA-Factory environment.

See [LLaMA-Factory Docs](https://github.com/hiyouga/LlamaFactory)

```bash
conda create -n lf python=3.11
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```

**We recommend using LLaMA-Factory** (`llamafactory-cli api`) rather than pure vLLM (`vllm serve`).
Although vLLM can also be used for deployment, we observed slightly lower performance compared to LLaMA-Factory, likely due to differences in chat template handling.

1. Run Deploy Script

In directory `curious_vla/navsim_eval`:

```bash
model_path=/path/to/model
num_instances=num_of_gpus_per_node # default is 8, for an 8xGPU node
tmux new-session -d -s vllm_eval "bash -c 'model_name_or_path=$model_path template=qwen2_vl bash lf_serve_cot.sh $num_instances 2>&1 | tee /tmp/vllm_eval.log'"
```

After this, the server will be ready.

If want to use `vllm serve`, see `navsim_eval/vllm_serve.sh`.

## 2. Prompt & Eval (Client)

1. **Env2: navsim**: Create Navsim environment.

See [navsim_eval/README.md](../navsim_eval/README.md).

Steps:
- Build conda env
- Download data
- **Set env variables**
- **Build metric cache**:

```bash
cd navsim_eval && bash scripts/evaluation/run_metric_caching.sh
```

2. Run Eval Script

```bash
cd navsim_eval && bash scripts/evaluation/run_qwen_pdm_score_evaluation.sh
```

3. Prompt Construction

See `navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py`
