#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash examples/ablations/run_qwen3-8b_vat_opd.sh [--dry-run] MODEL GROUP [Hydra overrides...]
MODEL: dflash | dspark
GROUP: core | replay | replay_budget | front | planner | reject | anchors | window | temperature | sampling | legacy | all
       or one case name shown by --list
Environment: SPECO_ABLATION_SEEDS="42 43 44", SPECO_ABLATION_OUTPUT=/path/to/results
Model/data environment variables and GPU/TP overrides follow the base examples.
--dry-run prints shell-escaped commands without creating files or starting training.
--list lists cases. Each real invocation runs its cases sequentially and stops on failure.
EOF
}

all_cases=(baseline vat opd opd_truncated opd_replay8 opd_mixed opd_reject2 opd_anchors64 opd_window512
    opd_t06 opd_prob_t1 opd_prob_t06 baseline_prob baseline_legacy opd_legacy)
dry_run=false
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ ${1:-} == --list ]]; then
    printf '%s\n' "${all_cases[@]}" vat_front
    exit 0
fi
if [[ ${1:-} == --dry-run ]]; then
    dry_run=true
    shift
fi
if (( $# < 2 )); then
    usage >&2
    exit 2
fi
model=$1
group=$2
shift 2
case "$model" in
    dflash|dspark) ;;
    *) echo "Unknown model: $model" >&2; exit 2 ;;
esac
case "$group" in
    core) cases=(baseline vat opd) ;;
    replay) cases=(opd_truncated opd) ;;
    replay_budget) cases=(opd opd_replay8) ;;
    front) cases=(vat vat_front) ;;
    planner) cases=(opd_truncated opd_mixed) ;;
    reject) cases=(opd opd_reject2) ;;
    anchors) cases=(opd_truncated opd_anchors64) ;;
    window) cases=(opd opd_window512) ;;
    temperature) cases=(opd opd_t06 opd_prob_t1 opd_prob_t06) ;;
    sampling) cases=(baseline baseline_prob opd_t06 opd_prob_t06) ;;
    legacy) cases=(baseline_legacy opd_legacy baseline opd) ;;
    all)
        cases=("${all_cases[@]}")
        if [[ $model == dflash ]]; then cases+=(vat_front); fi
        ;;
    *)
        cases=()
        for candidate in "${all_cases[@]}" vat_front; do
            if [[ $group == "$candidate" ]]; then cases=("$candidate"); break; fi
        done
        if (( ${#cases[@]} == 0 )); then echo "Unknown group/case: $group" >&2; exit 2; fi
        ;;
esac
if [[ $model == dspark && " ${cases[*]} " == *" vat_front "* ]]; then
    echo "front/vat_front is only defined for DFlash." >&2
    exit 2
fi
read -r -a seeds <<< "${SPECO_ABLATION_SEEDS:-42 43 44}"
if (( ${#seeds[@]} == 0 )); then echo "At least one seed is required." >&2; exit 2; fi
for seed in "${seeds[@]}"; do
    if [[ ! $seed =~ ^[0-9]+$ ]]; then echo "Invalid seed: $seed" >&2; exit 2; fi
done

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$repo_root"
output_root=${SPECO_ABLATION_OUTPUT:-$repo_root/outputs/vat-opd/$(date +%Y%m%d_%H%M%S)}
rollout=actor_rollout_ref.rollout
training=$rollout.drafter.training
opd=$training.drafter_opd
launcher="examples/run_qwen3-8b-drafter_opd_${model}_vllm.sh"

common=(
    "$rollout.temperature=0.6"
    "$rollout.top_p=1.0"
    "$rollout.top_k=-1"
    "++$rollout.engine_kwargs.vllm.async_scheduling=false"
    "$rollout.drafter.vllm.draft_sample_method=greedy"
    "$training.${model}_loss_mode=full_vocab"
    "$training.${model}_num_anchors=256"
    "$training.${model}_max_window=2048"
    "$training.${model}_per_position_loss_weight=fixed_exp_decay"
    "$training.lk_temperature=1.0"
    "$training.collect_interval_steps=2"
    "$training.training_interval_steps=2"
    "$training.step=10"
    "$training.publish_async=false"
    "$training.scheduler.execution.strategy=sync"
    "$training.packing.enable=false"
    "$training.use_data_buffer=false"
    "$training.sample_last_n_steps=1"
    "$training.hidden_state_window_tokens_per_sample=2048"
    "$opd.enable=true"
    "$opd.capture_verify_trace=true"
    "$opd.replay_full_suffix=true"
    "$opd.anchor_planner=trace"
    "$opd.random_anchor_ratio=0.25"
    "$opd.first_reject_weight=1.0"
    "$opd.loss=e2e_tv"
    "data.max_response_length=2048"
    "trainer.total_training_steps=200"
    "trainer.total_epochs=6"
    "trainer.resume_mode=disable"
    'trainer.logger=["console"]'
    "trainer.save_freq=100"
    "trainer.test_freq=-1"
)
if [[ $model == dflash ]]; then
    common+=("$training.dflash_front_position_weight=1.0" "$training.dflash_front_position_count=0")
else
    common+=("$training.dspark_confidence_loss_alpha=0.0" "$training.dspark_confidence_head_alpha=0.0")
fi

for seed in "${seeds[@]}"; do
    for experiment in "${cases[@]}"; do
        variant=()
        case "$experiment" in
            baseline|baseline_prob|baseline_legacy)
                variant+=("$opd.enable=false" "$opd.capture_verify_trace=false" "$opd.replay_full_suffix=false") ;;
            vat|vat_front)
                variant+=("$opd.enable=false" "$opd.capture_verify_trace=false" "$opd.replay_full_suffix=false"
                    "$training.${model}_per_position_loss_weight=vat"
                    "$training.vat.accept_mode=greedy" "$training.vat.self_cov_weight=0.0")
                if [[ $model == dspark ]]; then variant+=("$training.dspark_confidence_head_alpha=1.0"); fi
                if [[ $experiment == vat_front ]]; then
                    variant+=("$training.dflash_front_position_weight=2.0" "$training.dflash_front_position_count=3")
                fi
                ;;
            opd_truncated) variant+=("$opd.replay_full_suffix=false") ;;
            opd_replay8) variant+=("$opd.replay_max_blocks_per_sample=8") ;;
            opd_mixed) variant+=("$opd.anchor_planner=mixed" "$opd.replay_full_suffix=false") ;;
            opd_reject2) variant+=("$opd.first_reject_weight=2.0") ;;
            opd_anchors64) variant+=("$training.${model}_num_anchors=64" "$opd.replay_full_suffix=false") ;;
            opd_window512) variant+=("$training.${model}_max_window=512" "$training.hidden_state_window_tokens_per_sample=512") ;;
        esac
        case "$experiment" in
            opd_t06|opd_prob_t06|opd_legacy) variant+=("$training.lk_temperature=0.6") ;;
        esac
        case "$experiment" in
            baseline_prob|opd_prob_t1|opd_prob_t06)
                variant+=("$rollout.drafter.vllm.draft_sample_method=probabilistic") ;;
        esac
        case "$experiment" in
            baseline_legacy|opd_legacy)
                variant+=("$training.${model}_num_anchors=64" "$training.${model}_max_window=512" "$training.hidden_state_window_tokens_per_sample=512") ;;
        esac
        if [[ $experiment == opd_legacy ]]; then
            variant+=("$opd.anchor_planner=mixed" "$opd.first_reject_weight=2.0" "$opd.replay_full_suffix=false")
        fi

        run_name="${model}_${experiment}_seed${seed}"
        run_dir="$output_root/$run_name"
        # Seeds and run identity are reserved; other user overrides take precedence.
        cmd=(bash "$launcher" "${common[@]}" "${variant[@]}" "$@"
            "data.seed=$seed" "actor_rollout_ref.actor.data_loader_seed=$seed"
            "++actor_rollout_ref.actor.fsdp_config.seed=$seed" "++$rollout.engine_kwargs.vllm.seed=$seed"
            "trainer.experiment_name=$run_name" "trainer.default_local_dir=$run_dir/checkpoints")
        if $dry_run; then
            printf '%q ' "${cmd[@]}"
            printf '\n'
            continue
        fi
        # Refuse to mix checkpoints/logs with a previous run.
        mkdir -p -- "$output_root"
        mkdir -- "$run_dir"
        printf '%q ' "${cmd[@]}" > "$run_dir/command.sh"
        printf '\n' >> "$run_dir/command.sh"
        git rev-parse HEAD > "$run_dir/revision.txt"
        git status --short >> "$run_dir/revision.txt"
        echo "Running $run_name -> $run_dir"
        "${cmd[@]}" 2>&1 | tee "$run_dir/train.log"
    done
done
