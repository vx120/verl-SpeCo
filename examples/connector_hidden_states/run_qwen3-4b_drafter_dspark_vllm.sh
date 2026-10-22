#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${TARGET_MODEL:?Set TARGET_MODEL to the Qwen3-4B directory}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL to the DSpark checkpoint directory}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the input parquet file}"
RUN=${OUTPUT_DIR:-$REPO/outputs/connector-cotrain}
PYTHON_BIN=${PYTHON_BIN:-$REPO/.venv/bin/python}
mkdir -p "$RUN"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
transport=()
case "${SPECO_HIDDEN_TRANSPORT:-ordinary}" in
  ordinary) ;;
  mooncake)
    : "${MOONCAKE_MASTER_ADDRESS:?Start a dedicated master and set host:port}"
    transport=(
      actor_rollout_ref.rollout.drafter.training.transfer_queue.enable=true
      "actor_rollout_ref.rollout.drafter.training.transfer_queue.run_id=${SPECO_RUN_ID:-connector-$(date +%s)-$$}"
      actor_rollout_ref.rollout.drafter.training.transfer_queue.backend.storage_backend=MooncakeStore
      ++actor_rollout_ref.rollout.drafter.training.transfer_queue.backend.MooncakeStore.auto_init=false
      ++actor_rollout_ref.rollout.drafter.training.transfer_queue.backend.MooncakeStore.metadata_server=P2PHANDSHAKE
      "++actor_rollout_ref.rollout.drafter.training.transfer_queue.backend.MooncakeStore.master_server_address=$MOONCAKE_MASTER_ADDRESS"
      "++actor_rollout_ref.rollout.drafter.training.transfer_queue.backend.MooncakeStore.local_hostname=${MOONCAKE_LOCAL_HOSTNAME:-127.0.0.1}"
      ++actor_rollout_ref.rollout.drafter.training.transfer_queue.backend.MooncakeStore.protocol=tcp
      ++actor_rollout_ref.rollout.drafter.training.transfer_queue.backend.MooncakeStore.global_segment_size=67108864
      ++actor_rollout_ref.rollout.drafter.training.transfer_queue.backend.MooncakeStore.local_buffer_size=16777216
      ++actor_rollout_ref.rollout.drafter.training.transfer_queue.backend.MooncakeStore.use_gdr=false
    ) ;;
  *) echo 'SPECO_HIDDEN_TRANSPORT must be ordinary or mooncake' >&2; exit 2 ;;
esac
export HYDRA_FULL_ERROR=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export VERL_LOGGING_LEVEL=INFO PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" -m verl_speco.main \
 algorithm.adv_estimator=grpo \
 ray_kwargs.ray_init.num_cpus=16 \
 +ray_kwargs.ray_init._system_config.prestart_worker_first_driver=false \
 +ray_kwargs.ray_init._system_config.num_workers_soft_limit=8 \
 "data.train_files=$TRAIN_DATA" \
 "data.val_files=${VAL_DATA:-$TRAIN_DATA}" \
 data.train_batch_size=2 data.max_prompt_length=128 data.max_response_length=64 \
 data.filter_overlong_prompts=false data.truncation=left data.shuffle=false \
 "actor_rollout_ref.model.path=$TARGET_MODEL" \
 actor_rollout_ref.model.use_remove_padding=true \
 actor_rollout_ref.model.enable_gradient_checkpointing=true \
 actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.strategy=fsdp2 \
 actor_rollout_ref.actor.ppo_mini_batch_size=2 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.actor.calculate_entropy=false \
 actor_rollout_ref.actor.fsdp_config.param_offload=true \
 actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
 actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=2 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.agent.num_workers=2 \
 actor_rollout_ref.rollout.temperature=0.6 actor_rollout_ref.rollout.enforce_eager=true \
 actor_rollout_ref.rollout.enable_chunked_prefill=false \
 actor_rollout_ref.rollout.enable_prefix_caching=false \
 actor_rollout_ref.rollout.max_model_len=256 \
 actor_rollout_ref.rollout.max_num_seqs=2 \
 actor_rollout_ref.rollout.max_num_batched_tokens=512 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=TRITON_ATTN \
 +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=false \
 actor_rollout_ref.rollout.drafter.enable=true \
 actor_rollout_ref.rollout.drafter.enable_drafter_training=true \
 "actor_rollout_ref.rollout.drafter.model_path=$DRAFT_MODEL" \
 actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
 actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_vllm=true \
 actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=false \
 actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=false \
 actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=32 \
 actor_rollout_ref.rollout.drafter.training.hidden_state_window_min_rows=8 \
 actor_rollout_ref.rollout.drafter.training.vllm_hidden_state_window.min_rows=8 \
 actor_rollout_ref.rollout.drafter.training.dspark_block_size=7 \
 actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=2 \
 actor_rollout_ref.rollout.drafter.training.dspark_max_window=32 \
 actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=restricted_ce \
 actor_rollout_ref.rollout.drafter.training.target_lm_head_row_restricted_sync=false \
 actor_rollout_ref.rollout.drafter.training.dspark_ce_loss_alpha=0.1 \
 actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=0.45 \
 actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha=0 \
 actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.drafter.training.step=1 \
 actor_rollout_ref.rollout.drafter.training.train_batches_per_cycle=1 \
 actor_rollout_ref.rollout.drafter.training.collect_interval_steps=1 \
 actor_rollout_ref.rollout.drafter.training.training_interval_steps=1 \
 actor_rollout_ref.rollout.drafter.training.publish_async=false \
 actor_rollout_ref.rollout.drafter.training.publish_dtype=bf16 \
 actor_rollout_ref.rollout.drafter.training.draft_update_weights_bucket_megabytes=64 \
 actor_rollout_ref.rollout.drafter.training.draft_update_flush_before=false \
 actor_rollout_ref.rollout.drafter.training.draft_update_flush_after=true \
 actor_rollout_ref.rollout.drafter.rollout.spec_steps=1 \
 actor_rollout_ref.rollout.drafter.rollout.spec_topk=1 \
 actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=7 \
 algorithm.use_kl_in_reward=false trainer.val_before_train=false \
 trainer.logger='[console]' trainer.project_name=hidden_state_comparison \
 trainer.experiment_name=connector_cotrain_train_only \
 trainer.resume_mode=disable trainer.n_gpus_per_node=2 trainer.nnodes=1 \
 trainer.total_training_steps=2 trainer.total_epochs=1 \
 trainer.save_freq=-1 trainer.test_freq=-1 \
 "actor_rollout_ref.rollout.drafter.checkpoint_path=$RUN/checkpoints/drafter" \
 ++actor_rollout_ref.rollout.drafter.training.vllm_hidden_capture_backend=connector \
 ++actor_rollout_ref.rollout.drafter.training.connector_train_only=true \
 "++actor_rollout_ref.rollout.drafter.training.connector_export_path=$RUN/exports" \
 ++actor_rollout_ref.rollout.drafter.training.connector_target_layer_ids='[2,10,18,26,34]' \
 "trainer.default_local_dir=$RUN/checkpoints" "${transport[@]}" "$@"
