#!/usr/bin/env bash
# Synchronous OPD + online DFlash co-training | vLLM rollout | FSDP2 | NVIDIA GPUs

set -xeuo pipefail

cd /model/ljh/opd_co-train/verl-SpeCo

export CUDA_VISIBLE_DEVICES=4,5,6
export RAY_DEDUP_LOGS=0
export VERL_SPECO_VLLM_DRAFT_DIAG=1

STUDENT_MODEL=${STUDENT_MODEL:-/nas/disk1/Qwen3-4B}
TEACHER_MODEL=${TEACHER_MODEL:-/nas/disk1/Qwen3-8B}
DRAFTER_MODEL=${DRAFTER_MODEL:-/model/ljh/model/dspark_qwen3_4b_block7}
TRAIN_FILES=${TRAIN_FILES:-/nas/disk6/ls/dataset/DAPO-Math-17k/data/dapo-math-17k.filtered_512.parquet}
VAL_FILES=${VAL_FILES:-/nas/disk6/ls/dataset/AIME_2024/aime-2024.parquet}
# CHECKPOINT_DIR=${CHECKPOINT_DIR:-/path/to/checkpoints}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
ROLLOUT_TP=${ROLLOUT_TP:-2}
TEACHER_TP=${TEACHER_TP:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
DRAFTER_TRAIN_BATCH_SIZE=${DRAFTER_TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-5120}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))


RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CKPTS_DIR=/model/ljh/opd_co-train/ckpt/training/${RUN_TIMESTAMP}
ROLLOUT_DIR=/model/ljh/opd_co-train/ckpt/training/${RUN_TIMESTAMP}

mkdir -p "${CKPTS_DIR}" "${ROLLOUT_DIR}" || exit 1
echo "Checkpoint output: ${CKPTS_DIR}"
echo "Rollout output: ${ROLLOUT_DIR}"


PYTHONUNBUFFERED=1 python3 -m verl_speco.main \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.shuffle=True \
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
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_WORKERS} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.drafter.enable=True \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
    actor_rollout_ref.rollout.drafter.model_path="${DRAFTER_MODEL}" \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
    actor_rollout_ref.rollout.drafter.training.dspark_hard_sample_ratio=0.2 \
    actor_rollout_ref.rollout.drafter.training.dspark_block_size=7 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=64 \
    actor_rollout_ref.rollout.drafter.training.dspark_max_window=512 \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_decay_gamma=7 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_target_layers=5 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_hidden_layers=5 \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_rank=256 \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_head_type=vanilla \
    actor_rollout_ref.rollout.drafter.training.dspark_ce_loss_alpha=0.1 \
    actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=0.9 \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha=0.0 \
    actor_rollout_ref.rollout.drafter.training.lr=5e-6 \
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=${DRAFTER_TRAIN_BATCH_SIZE} \
    actor_rollout_ref.rollout.drafter.training.sample_last_n_steps=0 \
    actor_rollout_ref.rollout.drafter.training.use_data_buffer=True \
    actor_rollout_ref.rollout.drafter.training.mode=online \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=False \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True \
    actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook \
    actor_rollout_ref.rollout.drafter.training.use_logits=False \
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_mode=random \
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=512 \
    actor_rollout_ref.rollout.drafter.training.hidden_state_random_seed_by_step=True \
    actor_rollout_ref.rollout.drafter.training.step=5 \
    actor_rollout_ref.rollout.drafter.training.max_collect_samples_per_step_per_replica=64 \
    actor_rollout_ref.rollout.drafter.training.max_collect_tokens_per_step_per_replica=32768 \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=2 \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=2 \
    actor_rollout_ref.rollout.drafter.training.publish_interval_steps=2 \
    actor_rollout_ref.rollout.drafter.training.publish_async=False \
    actor_rollout_ref.rollout.drafter.training.draft_update_pause_generation=True \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_before=True \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_after=True \
    actor_rollout_ref.rollout.drafter.rollout.spec_steps=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_topk=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=7 \
    distillation.enabled=True \
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE} \
    distillation.nnodes=${NNODES} \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP} \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.8 \
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN} \
    distillation.distillation_loss.loss_mode=k1 \
    distillation.distillation_loss.topk=64 \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=True \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    trainer.balance_batch=True \
    trainer.resume_mode=disable \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.rollout_data_dir=${ROLLOUT_DIR} \
    trainer.logger='["console"]' \
    trainer.project_name=verl_speco_opd_cotrain \
    trainer.experiment_name=qwen3_8b_opd_dflash_vllm_fsdp2 \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.total_training_steps=100 \
    trainer.save_freq=45 \
    trainer.test_freq=500 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "$@"
