#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/run_qwen3-8b_drafter_eagle3_vllm.sh" "$@" \
  trainer.total_training_steps=3 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.drafter.training.collect_interval_steps=2 \
  actor_rollout_ref.rollout.drafter.training.training_interval_steps=2 \
  actor_rollout_ref.rollout.drafter.training.publish_interval_steps=2
