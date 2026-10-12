#!/usr/bin/env bash
set -uo pipefail

EXAMPLE_DIR=/model/ljh/opd_co-train/verl-SpeCo/examples

kill_ray_vllm() {
    echo "Stopping Ray processes..."
    ray stop -f >/dev/null 2>&1 || true

    echo "Stopping verl-SpeCo training processes..."
    pkill -TERM -f 'python3? -m verl_speco\.main' || true

    echo "Stopping vLLM server processes..."
    pkill -TERM -f 'python3?.*(vllm\.entrypoints|vllm serve)' || true

    echo "Waiting for processes and GPU memory to release..."
    sleep 180

    echo "Cleanup completed."
}

run_experiment() {
    local name="$1"
    local script="${EXAMPLE_DIR}/${name}.sh"
    local log="${EXAMPLE_DIR}/${name}.log"
    local rc

    echo
    echo "============================================================"
    echo "Starting experiment: ${name}"
    echo "Script: ${script}"
    echo "Log: ${log}"
    echo "============================================================"

    kill_ray_vllm

    if bash "$script" >"$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi

    echo "Experiment ${name} exited with code ${rc}"

    if (( rc != 0 )); then
        echo "Last 100 lines of log:"
        tail -n 100 "$log" || true
        return "$rc"
    fi
}

# sleep 2.1h

# dflash
# run_experiment "run_qwen3-8b_drafter_dflash_vllm"
# run_experiment "run_qwen3-8b-lk-loss_drafter_dflash_vllm"

# # eagle3
# run_experiment "run_qwen3-8b-adaptive-hybrid-lk_drafter_eagle3_vllm"
# run_experiment "run_qwen3-8b_drafter_eagle3_vllm"
# run_experiment "run_qwen3-8b-lk-loss_drafter_eagle3_vllm"

# # dspark
# run_experiment "run_qwen3-8b_drafter_dspark_vllm"
# run_experiment "run_qwen3-8b-lk-loss_drafter_dspark_vllm"

# opd
# run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2"
# run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_no_training"
# run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_1e-6_step10"
# run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_1e-6_step5"
# run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_2e-6_step5"

# run_experiment "opd_baseline"
# run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_step10"
# run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_step10_4e-6"
# run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_4e-6"
# run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_2222"

# run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_no_training_step200"
# run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_step15_1e-5"
# run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_step15_1e-6"
# run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_step15_interval8"
run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_step10_1e-5"
run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_step15_1e-5"
run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_step5_1e-5"
run_experiment "run_qwen3-8b_opd_dflash_vllm_fsdp2_step10_3e-5"

echo
echo "All experiments completed."