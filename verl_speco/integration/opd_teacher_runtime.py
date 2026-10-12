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
"""Defer OPD teacher inference so it can overlap online drafter training."""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from functools import wraps
from types import SimpleNamespace

from verl_speco.integration.opd_cotrain import opd_cotrain_enabled

DEFERRED_TEACHER_REQUEST_KEY = "_speco_opd_teacher_request"


async def compute_opd_teacher(self, requests, prompt_width, response_width):
    import torch
    from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

    async def compute(request):
        output = SimpleNamespace(
            extra_fields={},
            multi_modal_data=request["multi_modal_data"],
            mm_processor_kwargs=request["mm_processor_kwargs"],
        )
        await self._speco_original_compute_teacher_logprobs(
            output,
            prompt_ids=request["prompt_ids"],
            response_ids=request["response_ids"],
            validate=False,
            sample_kwargs=request["sample_kwargs"],
        )
        return _pad_teacher_outputs(
            output.extra_fields["teacher_ids"],
            output.extra_fields["teacher_logprobs"],
            prompt_width=prompt_width,
            response_width=response_width,
            prompt_length=len(request["prompt_ids"]),
            response_length=len(request["response_ids"]),
            pad_token_id=self.tokenizer.pad_token_id,
        )

    outputs = await asyncio.gather(*(compute(request) for request in requests))
    return {
        "teacher_ids": torch.cat([output[0] for output in outputs], dim=0),
        "teacher_logprobs": torch.cat([output[1] for output in outputs], dim=0),
    }


def install_opd_teacher_worker_patch(worker_cls):
    if getattr(worker_cls, "_speco_opd_teacher_patched", False):
        return
    original = worker_cls._compute_teacher_logprobs

    @wraps(original)
    async def compute_teacher_logprobs(
        self, output, prompt_ids, response_ids, validate, sample_kwargs=None
    ):
        cotrain = getattr(self, "_speco_opd_cotrain", False) or opd_cotrain_enabled(
            self.config
        )
        if cotrain and not validate:
            output.extra_fields[DEFERRED_TEACHER_REQUEST_KEY] = {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "multi_modal_data": output.multi_modal_data,
                "mm_processor_kwargs": output.mm_processor_kwargs,
                "sample_kwargs": {self.teacher_key: sample_kwargs.get(self.teacher_key)}
                if sample_kwargs is not None
                else None,
            }
            return
        await original(self, output, prompt_ids, response_ids, validate, sample_kwargs)

    worker_cls._speco_original_compute_teacher_logprobs = original
    worker_cls._compute_teacher_logprobs = compute_teacher_logprobs
    worker_cls.compute_opd_teacher = compute_opd_teacher
    worker_cls._speco_opd_teacher_patched = True


@contextmanager
def parallel_opd_teacher_inference(batch, workers, resolve):
    """Submit teacher RPCs, run drafter work, then join before student training."""
    import numpy as np
    import torch

    started = time.perf_counter()
    requests = batch.non_tensor_batch.pop(DEFERRED_TEACHER_REQUEST_KEY)
    chunks = np.array_split(requests, min(len(requests), len(workers)))
    refs = [
        worker.compute_opd_teacher.remote(
            chunk.tolist(),
            batch.batch["prompts"].shape[1],
            batch.batch["responses"].shape[1],
        )
        for worker, chunk in zip(workers, chunks)
    ]
    metrics = {}
    try:
        yield metrics
    finally:
        wait_started = time.perf_counter()
        outputs = resolve(refs)
        for key in ("teacher_ids", "teacher_logprobs"):
            batch.batch[key] = torch.cat([output[key] for output in outputs], dim=0)
        metrics["timing_s/opd_teacher_wait"] = time.perf_counter() - wait_started
        metrics["timing_s/opd_teacher_drafter"] = time.perf_counter() - started
