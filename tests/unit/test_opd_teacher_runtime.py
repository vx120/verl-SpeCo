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

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from verl_speco.integration.opd_teacher_runtime import (
    DEFERRED_TEACHER_REQUEST_KEY,
    install_opd_teacher_worker_patch,
    parallel_opd_teacher_inference,
)


def _config(*, opd=True, train=True):
    return {
        "distillation": {"enabled": opd},
        "actor_rollout_ref": {
            "rollout": {"drafter": {"enable": True, "enable_drafter_training": train}}
        },
    }


@pytest.fixture
def worker_class():
    class Worker:
        def __init__(self, config):
            self.config = config
            self.teacher_key = "teacher_route"
            self.tokenizer = SimpleNamespace(pad_token_id=0)
            self.calls = []

        async def _compute_teacher_logprobs(
            self, output, prompt_ids, response_ids, validate, sample_kwargs=None
        ):
            self.calls.append(
                (prompt_ids, response_ids, validate, sample_kwargs, output)
            )
            if not validate:
                tokens = torch.tensor(prompt_ids + response_ids).unsqueeze(-1)
                output.extra_fields["teacher_ids"] = tokens
                output.extra_fields["teacher_logprobs"] = -tokens.float()

    install_opd_teacher_worker_patch(Worker)
    return Worker


def _output():
    return SimpleNamespace(
        extra_fields={},
        multi_modal_data={"images": ["image"]},
        mm_processor_kwargs={"max_pixels": 64},
    )


def test_cotrain_defers_teacher_and_preserves_request(worker_class):
    worker = worker_class(_config())
    output = _output()
    asyncio.run(
        worker._compute_teacher_logprobs(
            output, [11, 12], [21], False, {"teacher_route": "math", "irrelevant": 42}
        )
    )
    assert worker.calls == []
    assert output.extra_fields[DEFERRED_TEACHER_REQUEST_KEY] == {
        "prompt_ids": [11, 12],
        "response_ids": [21],
        "multi_modal_data": output.multi_modal_data,
        "mm_processor_kwargs": output.mm_processor_kwargs,
        "sample_kwargs": {"teacher_route": "math"},
    }


@pytest.mark.parametrize(
    "opd,train,validate",
    [
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_other_modes_delegate_to_upstream(worker_class, opd, train, validate):
    worker = worker_class(_config(opd=opd, train=train))
    output = _output()
    asyncio.run(worker._compute_teacher_logprobs(output, [1], [2], validate))
    assert len(worker.calls) == 1
    assert DEFERRED_TEACHER_REQUEST_KEY not in output.extra_fields


def test_worker_patch_is_idempotent(worker_class):
    original = worker_class._speco_original_compute_teacher_logprobs
    patched = worker_class._compute_teacher_logprobs
    install_opd_teacher_worker_patch(worker_class)
    assert worker_class._compute_teacher_logprobs is patched
    assert worker_class._speco_original_compute_teacher_logprobs is original


def test_deferred_teacher_uses_original_tokens_routing_and_padding(
    worker_class, monkeypatch
):
    teacher_module = ModuleType("verl.experimental.teacher_loop.teacher_manager")

    def pad(
        ids,
        logprobs,
        *,
        prompt_width,
        response_width,
        prompt_length,
        response_length,
        pad_token_id,
    ):
        padding = (0, 0, prompt_width - prompt_length, response_width - response_length)
        return F.pad(ids, padding, value=pad_token_id).unsqueeze(0), F.pad(
            logprobs, padding
        ).unsqueeze(0)

    teacher_module._pad_teacher_outputs = Mock(side_effect=pad)
    monkeypatch.setitem(sys.modules, teacher_module.__name__, teacher_module)
    worker = worker_class(_config())
    requests = []
    for prompts, responses, route in [([9], [7, 8], "math"), ([5, 6], [4], "code")]:
        output = _output()
        asyncio.run(
            worker._compute_teacher_logprobs(
                output, prompts, responses, False, {"teacher_route": route}
            )
        )
        requests.append(output.extra_fields[DEFERRED_TEACHER_REQUEST_KEY])
    result = asyncio.run(worker.compute_opd_teacher(requests, 3, 4))
    assert result["teacher_ids"].squeeze(-1).tolist() == [
        [0, 0, 9, 7, 8, 0, 0],
        [0, 5, 6, 4, 0, 0, 0],
    ]
    assert torch.equal(result["teacher_logprobs"], -result["teacher_ids"].float())
    assert [call[3]["teacher_route"] for call in worker.calls] == ["math", "code"]
    assert worker.calls[0][4].multi_modal_data == {"images": ["image"]}
    assert worker.calls[0][4].mm_processor_kwargs == {"max_pixels": 64}
    assert teacher_module._pad_teacher_outputs.call_count == 2


def _batch(requests):
    values = np.empty(len(requests), dtype=object)
    values[:] = requests
    return SimpleNamespace(
        batch={
            "prompts": torch.zeros(len(requests), 3),
            "responses": torch.zeros(len(requests), 4),
        },
        non_tensor_batch={DEFERRED_TEACHER_REQUEST_KEY: values},
    )


@pytest.mark.parametrize("nworkers", [1, 2, 8])
def test_parallel_teacher_overlaps_drafter_and_preserves_reordered_rows(nworkers):
    teacher_started, drafter_started = Event(), Event()
    batch = _batch([3, 1, 2])

    def compute(requests, prompt_width, response_width):
        assert (prompt_width, response_width) == (3, 4)
        teacher_started.set()
        assert drafter_started.wait(5), "teacher ran synchronously before drafter"
        values = torch.tensor(requests).reshape(-1, 1, 1)
        return {"teacher_ids": values, "teacher_logprobs": -values.float()}

    with ThreadPoolExecutor(max_workers=nworkers) as pool:
        workers = [
            SimpleNamespace(
                compute_opd_teacher=SimpleNamespace(
                    remote=lambda *args: pool.submit(compute, *args)
                )
            )
            for _ in range(nworkers)
        ]
        with parallel_opd_teacher_inference(
            batch, workers, lambda refs: [ref.result() for ref in refs]
        ) as metrics:
            assert teacher_started.wait(5)
            assert "teacher_ids" not in batch.batch
            drafter_started.set()
        assert batch.batch["teacher_ids"].flatten().tolist() == [3, 1, 2]
        assert DEFERRED_TEACHER_REQUEST_KEY not in batch.non_tensor_batch
        assert (
            metrics["timing_s/opd_teacher_drafter"]
            >= metrics["timing_s/opd_teacher_wait"]
            >= 0
        )


def test_teacher_failure_propagates():
    worker = SimpleNamespace(
        compute_opd_teacher=SimpleNamespace(remote=lambda *args: "ref")
    )

    def resolve(refs):
        raise RuntimeError("teacher failed")

    with pytest.raises(RuntimeError, match="teacher failed"):
        with parallel_opd_teacher_inference(_batch([1]), [worker], resolve):
            pass


def test_drafter_failure_joins_teacher_before_propagating():
    batch = _batch([1])
    worker = SimpleNamespace(
        compute_opd_teacher=SimpleNamespace(remote=lambda *args: "ref")
    )
    resolve = Mock(
        return_value=[
            {
                "teacher_ids": torch.ones(1, 1, 1),
                "teacher_logprobs": torch.zeros(1, 1, 1),
            }
        ]
    )
    with pytest.raises(RuntimeError, match="drafter failed"):
        with parallel_opd_teacher_inference(batch, [worker], resolve):
            raise RuntimeError("drafter failed")
    resolve.assert_called_once_with(["ref"])
