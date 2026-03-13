#!/bin/bash

set -x

# ===== To modify =====
PROJECT_ROOT="/path/to/curious_vla"

MODEL_PATH="$PROJECT_ROOT/EasyR1/checkpoints/sft/your_model_name"
data_path=$PROJECT_ROOT/EasyR1/data/QA_navtrain_poutine_style_full
exp_name=qwen2_5_vl_3b_navsim_adas
active_data_path=$PROJECT_ROOT/EasyR1/checkpoints/adas/your_adas_exp_name/group_stats_filtered_0.1.txt

reward_function_path=$PROJECT_ROOT/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py

export EXP_NAME=$exp_name
export NAVSIM_STAT_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag

# load path
# load_path="YOUR_CHECKPOINT_TO_ADAS_INFER" # /path/to/your_checkpoint/global_step_x


python3 -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files=${data_path}@train \
    data.val_files=${data_path}@test \
    data.format_prompt=None \
    data.max_response_length=3072 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=15 \
    data.token_filter_file=${active_data_path} \
    # trainer.load_checkpoint_path=${load_path}