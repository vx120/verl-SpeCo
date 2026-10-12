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
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest


def _adapter():
    """Exercise the adapter without requiring the distributed verl runtime."""
    source = (
        Path(__file__).resolve().parents[2] / "verl_speco/trainer/speco_ray_trainer.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {
        "_speco_start_drafter_training",
        "_speco_finish_drafter_training",
        "_speco_train_drafter",
    }
    trainer = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SpecoRayPPOTrainer"
    )
    trainer.body = [
        n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    trainer.bases = []
    namespace = {
        "time": time,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "_speco_metric_float": lambda value: (
            float(value) if value is not None else None
        ),
    }
    exec(
        compile(ast.Module(body=[trainer], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["SpecoRayPPOTrainer"]()


def _outcome():
    return SimpleNamespace(
        trained=True,
        metrics={"drafter/train_worker_results_consistent": 1},
        raw_results=[
            {
                "dspark/ce_loss": 2.0,
                "drafter/memory_free_gib": 3.0,
                "drafter/memory_allocated_gib": 4.0,
            },
            {
                "dspark/ce_loss": 4.0,
                "drafter/memory_free_gib": 2.0,
                "drafter/memory_allocated_gib": 5.0,
            },
        ],
    )


def test_overlap_preserves_plan_and_aggregates_worker_outcome():
    trainer = _adapter()
    started, release = Event(), Event()
    plan = object()

    def execute(actual_plan):
        assert actual_plan is plan
        started.set()
        assert release.wait(5)
        return _outcome()

    trainer._speco_execute_drafter_training = execute
    pending = trainer._speco_start_drafter_training(plan)
    try:
        assert started.wait(5)
        assert not pending["future"].done()
    finally:
        release.set()
    trained, metrics = trainer._speco_finish_drafter_training(pending)
    assert trained
    assert metrics["drafter/train_worker_results_consistent"] == 1
    assert metrics["dspark/ce_loss"] == 3.0
    assert metrics["drafter/memory_free_gib"] == 2.0
    assert metrics["drafter/memory_allocated_gib"] == 5.0


def test_overlap_propagates_scheduler_failure():
    trainer = _adapter()

    def execute(plan):
        raise RuntimeError("preflight failed")

    trainer._speco_execute_drafter_training = execute
    pending = trainer._speco_start_drafter_training(object())
    with pytest.raises(RuntimeError, match="preflight failed"):
        trainer._speco_finish_drafter_training(pending)


def test_sync_training_uses_same_outcome_metrics():
    trainer = _adapter()
    trainer._speco_execute_drafter_training = lambda plan: _outcome()
    trained, metrics = trainer._speco_train_drafter(object())
    assert trained
    assert metrics["dspark/ce_loss"] == 3.0
    assert metrics["drafter/train_worker_results_consistent"] == 1
