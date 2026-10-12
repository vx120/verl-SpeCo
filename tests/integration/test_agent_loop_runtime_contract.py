# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from verl_speco.integration import agent_loop_runtime


def test_agent_loop_patch_supports_release_v080_llm_server_client(monkeypatch) -> None:
    class AgentLoopWorker:
        async def _compute_teacher_logprobs(self, *args, **kwargs):
            return None

        async def generate_sequences(self, batch):
            return batch

        async def _run_agent_loop(self, sampling_params, trajectory):
            return sampling_params, trajectory

        def _postprocess(self, inputs, input_non_tensor_batch=None, validate=False):
            return inputs, input_non_tensor_batch, validate

    class AgentLoopManager:
        def __init__(self):
            self.config = None

        async def generate_sequences(self, prompts):
            return prompts

    class LLMServerClient:
        async def generate(self, *, sampling_params):
            return sampling_params

    release_v080_module = SimpleNamespace(
        AgentLoopWorker=AgentLoopWorker,
        AgentLoopManager=AgentLoopManager,
        LLMServerClient=LLMServerClient,
    )
    monkeypatch.setattr(agent_loop_runtime, "_PATCHED", False)
    monkeypatch.setattr(
        agent_loop_runtime, "_load_agent_loop_module", lambda: release_v080_module
    )

    assert agent_loop_runtime.install_agent_loop_runtime_patch() is True
    assert LLMServerClient._speco_patched_generate is True
    assert AgentLoopWorker._speco_patched_generate_sequences is True
    assert AgentLoopManager._speco_patched_generate_sequences is True

    global_steps_token = agent_loop_runtime._CURRENT_GLOBAL_STEPS.set(17)
    try:
        sampling_params = asyncio.run(
            LLMServerClient().generate(sampling_params={"temperature": 1.0})
        )
    finally:
        agent_loop_runtime._CURRENT_GLOBAL_STEPS.reset(global_steps_token)
    assert sampling_params["_verl_global_steps"] == 17


def test_opd_vllm_manager_binds_explicit_teacher_worker(monkeypatch) -> None:
    import sys
    from unittest.mock import Mock
    from verl_speco.integration.opd_teacher_runtime import (
        install_opd_teacher_worker_patch,
    )

    class Worker:
        async def _compute_teacher_logprobs(self, *args, **kwargs):
            pass

    install_opd_teacher_worker_patch(Worker)
    remote = Mock(side_effect=lambda cls: cls)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(remote=remote))
    manager = SimpleNamespace(
        config={
            "distillation": {"enabled": True},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "drafter": {"enable": True, "enable_drafter_training": True},
                }
            },
        }
    )
    agent_loop_runtime._configure_speco_agent_loop_manager_instance(manager, Worker)
    assert (
        manager.agent_loop_workers_class.compute_opd_teacher
        is Worker.compute_opd_teacher
    )
    assert "compute_opd_teacher" in manager.agent_loop_workers_class.__dict__
    agent_loop_runtime._configure_speco_agent_loop_manager_instance(manager, Worker)
    remote.assert_called_once()


def test_opd_worker_defers_teacher_with_hidden_drafter_config(monkeypatch) -> None:
    import sys
    from verl_speco.integration.opd_teacher_runtime import (
        DEFERRED_TEACHER_REQUEST_KEY,
        install_opd_teacher_worker_patch,
    )

    class Worker:
        async def _compute_teacher_logprobs(self, *args, **kwargs):
            raise AssertionError("teacher should be deferred")

    install_opd_teacher_worker_patch(Worker)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(remote=lambda cls: cls))
    manager = SimpleNamespace(
        _speco_opd_cotrain=True,
        config={
            "distillation": {"enabled": True},
            "actor_rollout_ref": {"rollout": {"name": "vllm"}},
        },
    )
    agent_loop_runtime._configure_speco_agent_loop_manager_instance(manager, Worker)
    worker = manager.agent_loop_workers_class.__new__(manager.agent_loop_workers_class)
    worker.config = manager.config
    worker.teacher_key = "teacher"
    output = SimpleNamespace(
        extra_fields={}, multi_modal_data=None, mm_processor_kwargs=None
    )
    asyncio.run(worker._compute_teacher_logprobs(output, [1], [2], False))
    assert output.extra_fields[DEFERRED_TEACHER_REQUEST_KEY]["prompt_ids"] == [1]
