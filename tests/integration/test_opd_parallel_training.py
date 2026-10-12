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
"""Exercise the production fit hook without importing the GPU/Ray stack."""

from __future__ import annotations

import ast
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from verl_speco.integration.opd_cotrain import opd_cotrain_enabled
from verl_speco.integration.opd_teacher_runtime import (
    DEFERRED_TEACHER_REQUEST_KEY,
    parallel_opd_teacher_inference,
)
from verl_speco.trainer.scheduler import CollectionOutcome, CollectionPlan


@pytest.fixture
def trainer_class():
    # Load the actual hook, preserving its closures, without loading GPU workers.
    source = (
        Path(__file__).resolve().parents[2] / "verl_speco/trainer/speco_ray_trainer.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SpecoRayPPOTrainer"
    )
    hook = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "_speco_online_fit_hooks"
    )
    namespace = {
        "contextmanager": contextmanager,
        "MethodType": MethodType,
        "time": time,
        "DataProto": object,
        "Any": object,
        "CollectionPlan": CollectionPlan,
        "CollectionOutcome": CollectionOutcome,
        "opd_cotrain_enabled": opd_cotrain_enabled,
        "parallel_opd_teacher_inference": parallel_opd_teacher_inference,
        "_speco_metric_float": lambda value: None if value is None else float(value),
    }
    exec(
        compile(ast.Module(body=[hook], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return type("Trainer", (), {"_speco_online_fit_hooks": namespace[hook.name]})


@pytest.mark.parametrize("opd", [False, True])
@pytest.mark.parametrize("launch", [False, True])
@pytest.mark.parametrize("defer_publish", [False, True])
def test_fit_hook_orders_parallel_stage_actor_and_publish(
    trainer_class, opd, launch, defer_publish
):
    events = []
    teacher_started, drafter_done = Event(), Event()
    trainer = trainer_class()
    trainer.config = {
        "distillation": {"enabled": opd},
        "actor_rollout_ref": {
            "rollout": {"drafter": {"enable": True, "enable_drafter_training": True}}
        },
    }
    trainer.global_steps = 5
    trainer._pending_target_lm_head_sync = None
    trainer._speco_last_collected_samples = 1
    plan = SimpleNamespace(
        launch=launch,
        interval_matched=launch,
        reason="training_ready" if launch else "interval_not_reached",
    )
    requests = np.empty(1, dtype=object)
    requests[0] = "request"
    batch = SimpleNamespace(
        batch={"prompts": torch.zeros(1, 3), "responses": torch.zeros(1, 4)},
        non_tensor_batch={DEFERRED_TEACHER_REQUEST_KEY: requests} if opd else {},
    )

    def teacher(*args):
        teacher_started.set()
        if launch:
            assert drafter_done.wait(5), (
                "drafter must run while teacher RPC is in flight"
            )
        events.append("teacher_done")
        return {
            "teacher_ids": torch.ones(1, 7, 1),
            "teacher_logprobs": torch.zeros(1, 7, 1),
        }

    def prepare():
        events.append("prepare")
        trainer._pending_target_lm_head_sync = "head-ref" if launch else None
        return SimpleNamespace(training_plan=plan, metrics={})

    def train(plan):
        if opd:
            assert teacher_started.wait(5)
            assert "actor" not in events
        else:
            assert "actor" in events
        events.append("drafter")
        drafter_done.set()
        return True, {"drafter/trained": 1}

    def actor(batch):
        if opd:
            assert "teacher_done" in events
            assert batch.batch["teacher_ids"].shape == (1, 7, 1)
            assert DEFERRED_TEACHER_REQUEST_KEY not in batch.non_tensor_batch
            assert ("drafter" in events) is launch
        events.append("actor")
        return SimpleNamespace(meta_info={"metrics": {}})

    trainer._compute_old_log_prob = lambda batch: batch
    trainer._update_actor = actor
    trainer._speco_on_before_actor_update = prepare
    trainer._speco_log_drafter_training_plan = lambda plan: None
    trainer._speco_train_drafter = train
    trainer._speco_finish_target_lm_head_weight_sync = lambda pending: (
        events.append("sync_head") or {}
    )
    trainer._speco_get_drafter_runtime_state = lambda: SimpleNamespace(
        metrics=lambda: {}
    )
    trainer._speco_wait_pending_drafter_publish = lambda: None
    trainer._speco_oldlogprob_collection_requested = lambda: False
    trainer._speco_oldlogprob_entropy_hook_enabled = lambda: False
    trainer._ray_get_if_needed = lambda refs: [ref.result() for ref in refs]
    trainer._speco_update_output_metrics = lambda output, metrics: (
        output.meta_info["metrics"].update(metrics) or output
    )
    trainer._speco_publish_drafter_weights = lambda trained, plan, **kwargs: (
        events.append(("publish", trained)) or {}
    )
    if defer_publish:
        trainer.checkpoint_manager = SimpleNamespace(
            update_weights=lambda *args: events.append("update_weights")
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        rollout = SimpleNamespace(
            generate_sequences=lambda *args: None,
            agent_loop_workers=[
                SimpleNamespace(
                    compute_opd_teacher=SimpleNamespace(
                        remote=lambda *args: pool.submit(teacher, *args)
                    )
                )
            ],
        )
        trainer._speco_rollout_generation_target = lambda: rollout
        with trainer._speco_online_fit_hooks():
            output = trainer._update_actor(batch)
            if defer_publish and launch:
                assert not any(isinstance(event, tuple) for event in events)
                trainer.checkpoint_manager.update_weights(5)

    assert events[-1] == ("publish", launch)
    assert output.meta_info["metrics"]["drafter/trained"] == int(launch)
    assert events.count("drafter") == int(launch)
    if launch:
        assert events.index("sync_head") < events.index("drafter")
    if opd:
        assert output.meta_info["metrics"]["timing_s/opd_teacher_drafter"] >= 0
    else:
        assert "timing_s/opd_teacher_drafter" not in output.meta_info["metrics"]
    assert trainer._update_actor is actor
