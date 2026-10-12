from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _source_tree(relative_path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8-sig"))


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"missing function {name}")


def test_opd_example_uses_bounded_dspark_memory_defaults() -> None:
    script = (
        REPO_ROOT / "examples" / "run_qwen3-8b_opd_dflash_vllm_fsdp2.sh"
    ).read_text(encoding="utf-8")

    assert "DRAFTER_TRAIN_BATCH_SIZE=${DRAFTER_TRAIN_BATCH_SIZE:-8}" in script
    assert "dspark_num_anchors=64" in script
    assert "DRAFTER_L1_CHUNK_SIZE=${DRAFTER_L1_CHUNK_SIZE:-128}" in script
    assert (
        "DRAFTER_FULL_VOCAB_CHUNK_SIZE=${DRAFTER_FULL_VOCAB_CHUNK_SIZE:-128}" in script
    )
    assert "dspark_l1_chunk_size=${DRAFTER_L1_CHUNK_SIZE}" in script
    assert "dspark_full_vocab_chunk_size=${DRAFTER_FULL_VOCAB_CHUNK_SIZE}" in script
    assert "dspark_checkpoint_loss_chunks=True" in script
    assert "dspark_gradient_checkpointing=True" in script


def test_dspark_full_vocab_losses_are_checkpointed_in_chunks() -> None:
    tree = _source_tree("verl_speco/backends/dspark_trainer_backend.py")
    ce_function = _function(tree, "_compute_full_vocab_ce_for_active")
    l1_function = _function(tree, "_compute_l1_loss_for_active")

    def called_names(function: ast.AST) -> list[str]:
        return [
            node.func.attr
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]

    assert "_run_checkpointed_loss_chunk" in called_names(ce_function)
    assert "_run_checkpointed_loss_chunk" in called_names(l1_function)


def test_dspark_decoder_layers_use_activation_checkpointing() -> None:
    tree = _source_tree("verl_speco/models/dflash/modeling_dflash.py")
    draft_model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DFlashDraftModel"
    )
    forward = _function(draft_model, "forward")
    checkpoint_calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "checkpoint"
    ]

    assert len(checkpoint_calls) == 1
    use_reentrant = next(
        keyword.value
        for keyword in checkpoint_calls[0].keywords
        if keyword.arg == "use_reentrant"
    )
    assert isinstance(use_reentrant, ast.Constant)
    assert use_reentrant.value is False


def test_drafter_worker_samples_memory_around_training_and_cleanup() -> None:
    tree = _source_tree("verl_speco/workers/speco_worker.py")
    train_drafter = _function(tree, "train_drafter")
    calls = [
        node.func.attr
        for node in ast.walk(train_drafter)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]

    assert calls.count("reset_runtime_memory_stats") == 1
    assert calls.count("get_runtime_memory_metrics") == 4


def test_driver_aggregates_drafter_memory_metrics() -> None:
    tree = _source_tree("verl_speco/trainer/speco_ray_trainer.py")
    finish = _function(tree, "_speco_finish_drafter_training")
    string_values = {
        node.value for node in ast.walk(finish) if isinstance(node, ast.Constant)
    }

    assert "drafter/memory_" in string_values
    assert "_free_gib" in string_values
