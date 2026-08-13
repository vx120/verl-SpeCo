# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


def _upstream_repo_root() -> Path:
    upstream_root = os.getenv("VERL_SPECO_UPSTREAM_ROOT")
    if not upstream_root:
        pytest.skip("set VERL_SPECO_UPSTREAM_ROOT to check the release/v0.8.0 OPD flow")
    base = Path(upstream_root)
    for candidate in (base, base / "verl"):
        if (candidate / "verl").is_dir():
            return candidate
    raise AssertionError(
        "VERL_SPECO_UPSTREAM_ROOT must point to the upstream verl checkout "
        "or to a directory containing it"
    )


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"missing function {name}")


def _source_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def test_teacher_scores_final_student_prompt_and_response_only() -> None:
    root = _upstream_repo_root()
    tree = _source_tree(root / "verl" / "experimental" / "agent_loop" / "agent_loop.py")
    function = _function(tree, "_compute_teacher_logprobs")

    sequence_values = [
        keyword.value
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compute_teacher_logprobs_single"
        for keyword in node.keywords
        if keyword.arg == "sequence_ids"
    ]

    assert len(sequence_values) == 1
    sequence = sequence_values[0]
    assert isinstance(sequence, ast.BinOp) and isinstance(sequence.op, ast.Add)
    assert isinstance(sequence.left, ast.Name) and sequence.left.id == "prompt_ids"
    assert isinstance(sequence.right, ast.Name) and sequence.right.id == "response_ids"


def test_sync_trainer_reuses_oldlogprob_before_student_update() -> None:
    root = _upstream_repo_root()
    tree = _source_tree(root / "verl" / "trainer" / "ppo" / "ray_trainer.py")
    function = _function(tree, "fit")
    calls = [
        (node.func.attr, node.lineno)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"_compute_old_log_prob", "_update_actor"}
    ]
    oldlogprob_lines = [line for name, line in calls if name == "_compute_old_log_prob"]
    student_update_lines = [line for name, line in calls if name == "_update_actor"]

    assert len(oldlogprob_lines) == 1
    assert len(student_update_lines) == 1
    assert oldlogprob_lines[0] < student_update_lines[0]


def _called_names(function: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_drafter_distributed_paths_use_fsdp2_apis() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    tree = _source_tree(repo_root / "verl_speco" / "trainer" / "base_trainer.py")

    build_calls = _called_names(_function(tree, "_build_draft_model"))
    publish_calls = _called_names(_function(tree, "_get_trainable_state_dict"))
    checkpoint_calls = _called_names(_function(tree, "_save_optimizer_checkpoint"))

    assert {"apply_fsdp2", "fsdp2_load_full_state_dict"} <= build_calls
    assert "get_fsdp_full_state_dict" in publish_calls
    assert "save" in checkpoint_calls


def _call_line(function: ast.AST, name: str) -> int:
    lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    ]
    assert len(lines) == 1, f"expected one call to {name}, got {len(lines)}"
    return lines[0]


def test_sync_drafter_lifecycle_is_wired_to_student_update_and_weight_sync() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    tree = _source_tree(repo_root / "verl_speco" / "trainer" / "speco_ray_trainer.py")
    hook = _function(tree, "_speco_online_fit_hooks")
    update_actor = _function(hook, "update_actor_with_speco")
    update_weights = _function(hook, "update_weights_with_speco")

    assert _call_line(update_actor, "original_update_actor") < _call_line(
        update_actor, "after_student_update"
    )
    assert _call_line(
        update_weights, "original_checkpoint_update_weights"
    ) < _call_line(update_weights, "after_student_rollout_weight_sync")
