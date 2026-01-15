#!/bin/bash
set -x

# RAPO Training Script for Qwen2.5-1.5B on DeepMath-103K
# Based on: https://arxiv.org/abs/2510.03865
# Key features:
# 1. Forward KL divergence (rapo_kl) instead of reverse KL
# 2. RAPO advantage estimator (similar to GRPO)
# 3. Rule-based reward (1 for correct, 0 for incorrect)

# Dataset paths
deepmath_train_path=$HOME/data/deepmath103k/train.parquet
deepmath_test_path=$HOME/data/deepmath103k/test.parquet

train_files="['$deepmath_train_path']"
test_files="['$deepmath_test_path']"

# Model path
MODEL_PATH=/home/yliog/model/qwen1.5b

# Disable wandb (use offline mode)
export WANDB_MODE=offline

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rapo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=rapo_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_rapo_deepmath' \
    trainer.experiment_name='qwen1_5b_rapo_deepmath103k' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 $@
