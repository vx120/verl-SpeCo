# Copyright 2026 Individual Contributor: vx120
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

import asyncio
import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl_speco.integration.vllm_hidden_capture import (
    _resolve_aux_layer_ids,
    activate_vllm_hidden_capture,
    build_vllm_hidden_capture_context,
    install_vllm_hidden_capture_adapter,
    vllm_capture_to_drafter_sample,
)


@dataclass
class _FakePlan:
    request_id: str
    prompt_len: int
    window_start_abs: int
    window_end_abs: int
    collection_id: str
    min_rows: int
    aux_layer_ids: tuple[int, ...]
    hidden_layout: str
    logprob_topk: int = 0

    @classmethod
    def from_window(
        cls,
        request_id,
        prompt_len,
        start,
        end,
        collection_id,
        *,
        coordinate,
        min_rows,
        aux_layer_ids,
        hidden_layout,
        logprob_topk=0,
    ):
        offset = prompt_len - 1 if coordinate == "response" else 0
        return cls(
            request_id,
            prompt_len,
            start + offset,
            end + offset,
            collection_id,
            min_rows,
            aux_layer_ids,
            hidden_layout,
            logprob_topk,
        )


def _install_fake_vllm(monkeypatch) -> None:
    module = types.ModuleType("vllm.v1.hidden_state_capture")
    module.HiddenStateCapturePlan = _FakePlan
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.hidden_state_capture", module)
    aux_module = types.ModuleType("vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils")
    aux_module.get_eagle3_aux_layers_from_config = lambda config: None
    monkeypatch.setitem(sys.modules, aux_module.__name__, aux_module)


def _engine(layer_ids=(), method="eagle3"):
    speculative = SimpleNamespace(method=method, target_layer_ids=layer_ids)
    return SimpleNamespace(vllm_config=SimpleNamespace(speculative_config=speculative))


def test_dspark_aux_layer_ids_use_vllm_runner_semantics() -> None:
    engine = _engine((7, 15, 23), method="dspark")

    layer_ids = _resolve_aux_layer_ids(
        engine,
        get_aux_layers=lambda config: tuple(
            layer_id + 1 for layer_id in config.target_layer_ids
        ),
    )

    assert layer_ids == (8, 16, 24)


def _config(**training_overrides):
    training = {
        "collect_hidden_states_from_vllm": True,
        "collect_interval_steps": 1,
        "collection_sample_rate": 1.0,
        "hidden_state_window_tokens_per_sample": 4,
        "hidden_state_window_min_rows": 2,
        "vllm_hidden_state_window": {
            "mode": "fixed",
            "coordinate": "response",
            "start": 2,
            "end": 6,
            "min_rows": 2,
        },
    }
    training.update(training_overrides)
    return {"speculative_algorithm": "EAGLE3", "training": training}


def test_context_local_adapter_only_marks_the_active_request() -> None:
    seen = []

    class Engine:
        async def generate(self, *args, **kwargs):
            seen.append(kwargs.copy())
            yield SimpleNamespace(outputs=[])

    engine = Engine()
    install_vllm_hidden_capture_adapter(engine)
    context = SimpleNamespace(
        request_id="selected",
        plan="plan",
        result=None,
        skip_reason=None,
        output_token_ids=[],
    )

    async def run():
        with activate_vllm_hidden_capture(engine, context):
            async for _ in engine.generate(request_id="selected"):
                pass
            async for _ in engine.generate(request_id="other"):
                pass

    asyncio.run(run())
    assert seen[0]["hidden_state_capture"] == "plan"
    assert "hidden_state_capture" not in seen[1]


def test_adapter_install_failure_skips_capture_without_breaking_rollout() -> None:
    class Engine:
        __slots__ = ()

        async def generate(self, *args, **kwargs):
            yield SimpleNamespace(outputs=[])

    engine = Engine()
    context = SimpleNamespace(skip_reason=None)

    with activate_vllm_hidden_capture(engine, context):
        pass

    assert context.skip_reason == "adapter_unavailable"


def test_fixed_response_window_converts_to_absolute_positions(monkeypatch) -> None:
    _install_fake_vllm(monkeypatch)
    context, reason = build_vllm_hidden_capture_context(
        engine=_engine((3, 7)),
        drafter_config=_config(),
        prompt_ids=[10, 11, 12],
        sampling_params={"max_tokens": 10},
        request_id="req",
        global_step=5,
        replica_rank=1,
    )

    assert reason is None
    assert context is not None
    assert (context.plan.window_start_abs, context.plan.window_end_abs) == (4, 8)
    assert context.plan.aux_layer_ids == (3, 7)


def test_dspark_layout_matches_training_contract(monkeypatch) -> None:
    _install_fake_vllm(monkeypatch)
    config = _config(dspark_l1_loss_alpha=0.9)
    config["speculative_algorithm"] = "DSPARK"

    context, reason = build_vllm_hidden_capture_context(
        engine=_engine((3, 7), method="dspark"),
        drafter_config=config,
        prompt_ids=[10, 11, 12],
        sampling_params={"max_tokens": 10},
        request_id="req",
        global_step=1,
        replica_rank=0,
    )

    assert reason is None
    assert context is not None
    assert context.plan.hidden_layout == "aux_final"
    assert context.training_hidden_layout == "dflash_aux_plus_last"


@pytest.mark.parametrize(
    "algorithm,alpha,wire_layout,training_layout",
    [
        ("DSPARK", 0.9, "aux_final", "dflash_aux_plus_last"),
        ("DSPARK", 0.0, "dflash_aux", "dflash_aux"),
        ("EAGLE3", 0.9, "aux_final", "eagle3_aux_plus_last"),
        ("DFLASH", 0.9, "dflash_aux", "dflash_aux"),
    ],
)
def test_real_runner_plan_and_training_sample_layouts(
    monkeypatch, algorithm, alpha, wire_layout, training_layout
) -> None:
    """Exercise the sibling Runner contract without importing the CUDA engine."""
    source = (
        Path(__file__).resolve().parents[3]
        / "vllm-windowed-hidden-state-collection/vllm/v1/hidden_state_capture.py"
    )
    if os.environ.get("SPECO_RUNNER_CAPTURE_SOURCE"):
        source = Path(os.environ["SPECO_RUNNER_CAPTURE_SOURCE"])
    if not source.exists():
        pytest.skip("Sibling vLLM Runner checkout is required")
    spec = importlib.util.spec_from_file_location("_speco_runner_contract", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    _install_fake_vllm(monkeypatch)
    sys.modules[
        "vllm.v1.hidden_state_capture"
    ].HiddenStateCapturePlan = module.HiddenStateCapturePlan
    config = _config(dspark_l1_loss_alpha=alpha)
    config["speculative_algorithm"] = algorithm
    context, reason = build_vllm_hidden_capture_context(
        engine=_engine((3, 7), method=algorithm.lower()),
        drafter_config=config,
        prompt_ids=[10, 11, 12],
        sampling_params={"max_tokens": 10},
        request_id="req",
        global_step=1,
        replica_rank=0,
    )
    assert reason is None
    assert context.plan.hidden_layout == wire_layout
    buffer = module.HiddenStateCaptureBuffer(context.plan)
    width = 4 * (2 + (wire_layout == "aux_final"))
    buffer.add(
        module.HiddenStateCaptureChunk(
            np.array([4, 5]), torch.ones(2, width, dtype=torch.bfloat16)
        )
    )
    context.result = buffer.finish()
    sample = vllm_capture_to_drafter_sample(context, [20, 21, 22, 23])
    assert sample["hidden_states_layout"] == training_layout
    assert sample["hidden_states"].shape == (1, 2, width)


def test_short_generation_budget_fails_closed(monkeypatch) -> None:
    _install_fake_vllm(monkeypatch)
    context, reason = build_vllm_hidden_capture_context(
        engine=_engine((3,)),
        drafter_config=_config(),
        prompt_ids=[10, 11],
        sampling_params={"max_tokens": 3},
        request_id="req",
        global_step=1,
        replica_rank=0,
    )

    assert context is None
    assert reason == "generation_budget_too_short"


def test_result_becomes_position_aware_drafter_sample(monkeypatch) -> None:
    _install_fake_vllm(monkeypatch)
    context, _ = build_vllm_hidden_capture_context(
        engine=_engine((3,)),
        drafter_config=_config(),
        prompt_ids=[10, 11, 12],
        sampling_params={"max_tokens": 10},
        request_id="req",
        global_step=1,
        replica_rank=2,
    )
    assert context is not None
    context.result = SimpleNamespace(
        hidden_positions=np.array([4, 5]),
        hidden_states=torch.ones(2, 8, dtype=torch.bfloat16),
        hidden_window_start=4,
        hidden_window_end=8,
        hidden_layout="eagle3_aux_plus_last",
        collection_id="collection",
        copied_bytes=32,
        copy_ms=0.5,
    )

    sample = vllm_capture_to_drafter_sample(context, [20, 21, 22])

    assert sample is not None
    assert sample["hidden_states"].dtype == torch.bfloat16
    assert sample["hidden_positions"].tolist() == [[4, 5]]
    assert sample["input_ids"].tolist() == [[10, 11, 12, 20, 21, 22]]
    assert sample["replica_rank"] == 2


def test_use_logits_requests_window_topk(monkeypatch):
    _install_fake_vllm(monkeypatch)
    context, reason = build_vllm_hidden_capture_context(
        engine=_engine((1, 2, 3)),
        drafter_config=_config(use_logits=True, logits_topk=16),
        prompt_ids=[10, 20, 30],
        sampling_params={"max_tokens": 8},
        request_id="topk",
        global_step=1,
        replica_rank=0,
    )
    assert reason is None
    assert context.plan.logprob_topk == 16


def test_window_topk_supervision_keeps_predictor_positions():
    from verl_speco.integration.vllm_hidden_capture import VLLMHiddenCaptureContext

    plan = _FakePlan.from_window(
        "req",
        4,
        0,
        4,
        "test",
        coordinate="response",
        min_rows=1,
        aux_layer_ids=(),
        hidden_layout="final",
        logprob_topk=2,
    )
    logits = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    values, ids = logits.log_softmax(-1).topk(2, dim=-1)
    result = SimpleNamespace(
        hidden_positions=np.arange(3, 7),
        hidden_states=torch.ones(4, 2),
        hidden_window_start=3,
        hidden_window_end=7,
        collection_id="test",
        logprobs=values,
        logprob_token_ids=ids,
    )
    context = VLLMHiddenCaptureContext(
        request_id="req",
        plan=plan,
        prompt_ids=[1, 2, 3, 4],
        global_step=1,
        replica_rank=0,
        window_mode="front",
        result=result,
    )
    sample = vllm_capture_to_drafter_sample(context, [5, 6, 7, 8])
    values, ids = logits.log_softmax(-1).topk(2, dim=-1)
    assert sample["target_logprobs"].shape == (1, 4, 2, 2)
    torch.testing.assert_close(sample["target_logprobs"][0, ..., 0], values)
    assert torch.equal(sample["target_logprobs"][0, ..., 1].long(), ids)
    assert sample["target_logprobs_position_start"] == 3
    assert sample["target_logprobs_position_end"] == 7
    assert sample["hidden_raw_target_logprobs_positions"].tolist() == [[4, 5, 6, 7]]
    torch.testing.assert_close(
        sample["hidden_raw_target_logprobs"], sample["target_logprobs"]
    )
    context.result.logprobs = None
    assert vllm_capture_to_drafter_sample(context, [5, 6, 7, 8]) is None
