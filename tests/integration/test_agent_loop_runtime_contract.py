# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import asyncio
import ast
from pathlib import Path
import pytest
from types import SimpleNamespace

from verl_speco.integration import agent_loop_runtime


@pytest.mark.parametrize("validate,expected", [(False, 1), (True, 0)])
def test_vllm_capture_step_is_sent_before_generation(validate, expected):
    seen = []

    async def remote(step):
        seen.append(step)

    manager = SimpleNamespace(
        config={
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "drafter": {"training": {"collect_hidden_states_from_vllm": True}},
                }
            }
        },
        server_handles=[
            SimpleNamespace(set_speco_capture_step=SimpleNamespace(remote=remote))
        ],
    )
    prompts = SimpleNamespace(meta_info={"global_steps": 1, "validate": validate})
    asyncio.run(agent_loop_runtime._sync_vllm_capture_step(manager, prompts))
    assert seen == [expected]


def test_vllm_capture_step_does_not_touch_disabled_rollout():
    manager = SimpleNamespace(
        config={"actor_rollout_ref": {"rollout": {"name": "vllm"}}}
    )
    asyncio.run(agent_loop_runtime._sync_vllm_capture_step(manager, SimpleNamespace()))


def test_collection_wrapper_bridges_separate_server_manager():
    """Exercise the actual trainer closure with verl's split manager topology."""
    seen = []

    async def remote(step):
        seen.append(step)

    handle = SimpleNamespace(set_speco_capture_step=SimpleNamespace(remote=remote))
    manager = SimpleNamespace(
        config={
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "drafter": {"training": {"collect_hidden_states_from_vllm": True}},
                }
            }
        }
    )
    trainer = SimpleNamespace(
        llm_server_manager=SimpleNamespace(server_handles=[handle]),
        _speco_wait_pending_drafter_publish=lambda: None,
        _speco_store_rollout_metrics=lambda output: None,
        _speco_collect_generation_samples=lambda output: 0,
    )
    prompts = SimpleNamespace(meta_info={"global_steps": 1})

    def generate(batch):
        asyncio.run(agent_loop_runtime._sync_vllm_capture_step(manager, batch))
        return batch

    source = (
        Path(agent_loop_runtime.__file__).parents[1] / "trainer/speco_ray_trainer.py"
    )
    node = next(
        n
        for n in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef) and n.name == "generate_sequences_with_speco"
    )
    namespace = {
        "self": trainer,
        "original_generate_sequences": generate,
        "_speco_is_validation_generation": lambda *args: False,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    assert namespace["generate_sequences_with_speco"](manager, prompts) is prompts
    assert seen == [1]


def test_agent_loop_patch_supports_release_v080_llm_server_client(monkeypatch) -> None:
    class AgentLoopWorker:
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
