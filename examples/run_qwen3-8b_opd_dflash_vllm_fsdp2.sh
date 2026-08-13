#!/usr/bin/env bash
# Synchronous OPD + online DFlash co-training | vLLM rollout | FSDP2 | NVIDIA GPUs

set -xeuo pipefail

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-8B}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-32B}
DRAFTER_MODEL=${DRAFTER_MODEL:-/path/to/vllm-compatible-dflash-drafter}
TRAIN_FILES=${TRAIN_FILES:-/path/to/train.parquet}
VAL_FILES=${VAL_FILES:-/path/to/validation.parquet}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/path/to/checkpoints}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4}
ROLLOUT_TP=${ROLLOUT_TP:-2}
TEACHER_TP=${TEACHER_TP:-2}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-12288}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-12288}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}

MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))

PYTHONUNBUFFERED=1 python3 -m verl_speco.main \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.drafter.enable=True \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
    actor_rollout_ref.rollout.drafter.model_path="${DRAFTER_MODEL}" \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DFLASH \
    actor_rollout_ref.rollout.drafter.training.mode=online \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=False \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True \
    actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook \
    actor_rollout_ref.rollout.drafter.training.use_logits=False \
    actor_rollout_ref.rollout.drafter.training.step=10 \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=5 \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=5 \
    actor_rollout_ref.rollout.drafter.training.publish_async=False \
    actor_rollout_ref.rollout.drafter.training.draft_update_pause_generation=True \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_before=True \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_after=True \
    actor_rollout_ref.rollout.drafter.rollout.spec_steps=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_topk=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=16 \
    distillation.enabled=True \
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE} \
    distillation.nnodes=${NNODES} \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP} \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.4 \
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN} \
    distillation.distillation_loss.loss_mode=k1 \
    distillation.distillation_loss.topk=64 \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=True \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    trainer.balance_batch=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_speco_opd_cotrain \
    trainer.experiment_name=qwen3_8b_opd_dflash_vllm_fsdp2 \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "$@"
