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
from types import SimpleNamespace

from verl_speco.integration.agent_loop_runtime import (
    SPECO_TEACHER_SCORING_TIME_KEY,
    _compute_teacher_logprobs_with_timing,
)


def test_teacher_timing_delegates_upstream_arguments_unchanged() -> None:
    worker = SimpleNamespace(distillation_enabled=True)
    output = SimpleNamespace(extra_fields={})
    prompt_ids = [1, 2]
    response_ids = [3, 4]
    sample_kwargs = {"data_source": "math"}
    calls = []

    async def upstream(
        actual_worker,
        actual_output,
        actual_prompt_ids,
        actual_response_ids,
        actual_validate,
        actual_sample_kwargs,
    ):
        calls.append(
            (
                actual_worker,
                actual_output,
                actual_prompt_ids,
                actual_response_ids,
                actual_validate,
                actual_sample_kwargs,
            )
        )
        return "scored"

    result = asyncio.run(
        _compute_teacher_logprobs_with_timing(
            worker,
            upstream,
            output,
            prompt_ids,
            response_ids,
            False,
            sample_kwargs,
        )
    )

    assert result == "scored"
    assert calls == [(worker, output, prompt_ids, response_ids, False, sample_kwargs)]
    assert output.extra_fields[SPECO_TEACHER_SCORING_TIME_KEY] >= 0.0


def test_teacher_timing_is_not_recorded_for_validation() -> None:
    worker = SimpleNamespace(distillation_enabled=True)
    output = SimpleNamespace(extra_fields={})

    async def upstream(*args):
        return None

    asyncio.run(
        _compute_teacher_logprobs_with_timing(
            worker,
            upstream,
            output,
            [1],
            [2],
            True,
        )
    )

    assert SPECO_TEACHER_SCORING_TIME_KEY not in output.extra_fields
