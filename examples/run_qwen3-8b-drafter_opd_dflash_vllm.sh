#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root with the model/data environment variables from the base example.
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1
bash examples/run_qwen3-8b_drafter_dflash_vllm.sh \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=false \
    actor_rollout_ref.rollout.drafter.training.dflash_block_size=16 \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=15 \
    actor_rollout_ref.rollout.drafter.training.dflash_loss_mode=full_vocab \
    actor_rollout_ref.rollout.drafter.training.dflash_per_position_loss_weight=fixed_exp_decay \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.enable=true \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.capture_verify_trace=true \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.replay_full_suffix=true \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.anchor_planner=trace \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.random_anchor_ratio=0.25 \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.first_reject_weight=1.0 \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.loss=e2e_tv \
    actor_rollout_ref.rollout.drafter.training.lk_temperature=1.0 \
    actor_rollout_ref.rollout.drafter.vllm.draft_sample_method=greedy \
    actor_rollout_ref.rollout.drafter.training.dflash_num_anchors=256 \
    actor_rollout_ref.rollout.drafter.training.dflash_max_window=2048 \
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=2048 \
    actor_rollout_ref.rollout.drafter.training.packing.enable=false \
    actor_rollout_ref.rollout.drafter.training.use_data_buffer=false \
    actor_rollout_ref.rollout.drafter.training.sample_last_n_steps=1 \
    actor_rollout_ref.rollout.drafter.training.publish_async=false \
    actor_rollout_ref.rollout.drafter.training.scheduler.execution.strategy=sync \
    actor_rollout_ref.rollout.drafter.training.target_lm_head_row_restricted_sync=false \
    actor_rollout_ref.rollout.drafter.training.dflash_front_position_weight=1.0 \
    actor_rollout_ref.rollout.drafter.training.dflash_front_position_count=0 \
    trainer.experiment_name=qwen3_8b_dflash_draft_opd_gpu "$@"
