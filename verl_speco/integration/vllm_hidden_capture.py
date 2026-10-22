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

"""Request-scoped vLLM hidden-state collection for online drafter training."""

from __future__ import annotations

import contextvars
import hashlib
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from verl_speco.integration.oldlogprob_layer_ids import (
    resolve_drafter_hidden_states_layout,
)

_ACTIVE_CAPTURE: contextvars.ContextVar[VLLMHiddenCaptureContext | None] = (
    contextvars.ContextVar("speco_vllm_hidden_capture", default=None)
)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if hasattr(value, "get"):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_int_list(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return ()


def _stable_fraction(key: str) -> float:
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _stable_offset(key: str, inclusive_max: int) -> int:
    if inclusive_max <= 0:
        return 0
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (inclusive_max + 1)


def _sampling_max_tokens(sampling_params: Any) -> int | None:
    value = _get(sampling_params, "max_tokens", None)
    if value is None:
        value = _get(sampling_params, "max_new_tokens", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _resolve_aux_layer_ids(engine: Any, *, get_aux_layers=None) -> tuple[int, ...]:
    config = _get(engine, "vllm_config", None)
    speculative = _get(config, "speculative_config", None)
    if get_aux_layers is None:
        try:
            from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
                get_eagle3_aux_layers_from_config,
            )

            get_aux_layers = get_eagle3_aux_layers_from_config
        except ImportError:
            get_aux_layers = None
    if get_aux_layers is not None:
        layer_ids = _as_int_list(get_aux_layers(speculative))
        if layer_ids:
            return tuple(sorted(set(layer_ids)))
    candidates = [
        _get(speculative, "target_layer_ids", None),
        _get(speculative, "eagle_aux_hidden_state_layer_ids", None),
        _get(speculative, "target_hidden_layer_ids", None),
    ]
    draft_model_config = _get(speculative, "draft_model_config", None)
    hf_config = _get(draft_model_config, "hf_config", None)
    candidates.extend(
        [
            _get(hf_config, "target_layer_ids", None),
            _get(hf_config, "eagle_aux_hidden_state_layer_ids", None),
            _get(hf_config, "target_hidden_layer_ids", None),
        ]
    )
    for candidate in candidates:
        layer_ids = _as_int_list(candidate)
        if layer_ids:
            return layer_ids
    return ()


def _speculative_method(engine: Any, drafter_config: Any) -> str:
    config = _get(engine, "vllm_config", None)
    speculative = _get(config, "speculative_config", None)
    method = _get(speculative, "method", None)
    if method:
        return str(method).strip().lower()
    return str(_get(drafter_config, "speculative_algorithm", "")).strip().lower()


@dataclass
class VLLMHiddenCaptureContext:
    request_id: str
    plan: Any
    prompt_ids: list[int]
    global_step: int
    replica_rank: int
    window_mode: str
    training_hidden_layout: str = "final"
    result: Any | None = None
    skip_reason: str | None = None
    output_token_ids: list[int] = field(default_factory=list)


def install_vllm_hidden_capture_adapter(engine: Any) -> None:
    """Install one context-local wrapper around an AsyncLLM instance."""
    if bool(getattr(engine, "_speco_hidden_capture_wrapped", False)):
        return
    original_generate = engine.generate

    async def generate_with_hidden_capture(*args, **kwargs):
        active = _ACTIVE_CAPTURE.get()
        request_id = kwargs.get("request_id")
        if request_id is None and len(args) >= 3:
            request_id = args[2]
        if active is not None and str(request_id) == active.request_id:
            kwargs["hidden_state_capture"] = active.plan
        generator = original_generate(*args, **kwargs)
        if inspect.isawaitable(generator):
            generator = await generator
        async for item in generator:
            if active is not None and str(request_id) == active.request_id:
                capture = getattr(item, "hidden_state_capture", None)
                if capture is not None:
                    active.result = capture
                reason = getattr(item, "hidden_capture_skip_reason", None)
                if reason is not None:
                    active.skip_reason = str(reason)
                outputs = getattr(item, "outputs", None) or []
                if outputs:
                    active.output_token_ids = list(outputs[0].token_ids)
            yield item

    engine.generate = generate_with_hidden_capture
    engine._speco_hidden_capture_wrapped = True


@contextmanager
def activate_vllm_hidden_capture(
    engine: Any, context: VLLMHiddenCaptureContext | None
) -> Iterator[None]:
    if context is None:
        yield
        return
    try:
        install_vllm_hidden_capture_adapter(engine)
    except (AttributeError, TypeError):
        context.skip_reason = "adapter_unavailable"
        yield
        return
    token = _ACTIVE_CAPTURE.set(context)
    try:
        yield
    finally:
        _ACTIVE_CAPTURE.reset(token)


def build_vllm_hidden_capture_context(
    *,
    engine: Any,
    drafter_config: Any,
    prompt_ids: list[int],
    sampling_params: Any,
    request_id: str,
    global_step: int,
    replica_rank: int,
) -> tuple[VLLMHiddenCaptureContext | None, str | None]:
    """Build one fail-closed capture plan from the rollout training config."""
    training = _get(drafter_config, "training", {}) or {}
    if not bool(_get(training, "collect_hidden_states_from_vllm", False)):
        return None, "disabled"

    interval = int(_get(training, "collect_interval_steps", 1) or 1)
    if global_step <= 0 or interval <= 0 or global_step % interval:
        return None, "interval_not_reached"
    sample_rate = float(_get(training, "collection_sample_rate", 1.0) or 0.0)
    if sample_rate <= 0 or _stable_fraction(f"{global_step}:{request_id}") >= min(
        sample_rate, 1.0
    ):
        return None, "sample_rate"

    window = _get(training, "vllm_hidden_state_window", None) or {}
    mode = str(
        _get(window, "mode", _get(training, "hidden_state_window_mode", "front"))
        or "front"
    ).lower()
    coordinate = str(_get(window, "coordinate", "response") or "response").lower()
    min_rows = int(
        _get(window, "min_rows", _get(training, "hidden_state_window_min_rows", 1)) or 1
    )
    max_tokens = _sampling_max_tokens(sampling_params)
    default_rows = int(
        _get(training, "hidden_state_window_tokens_per_sample", 512) or 512
    )
    if mode == "fixed":
        start = int(_get(window, "start", 0) or 0)
        end = int(_get(window, "end", start + default_rows) or start + default_rows)
    elif mode in {"front", "random"}:
        width = max(default_rows, min_rows)
        if max_tokens is not None:
            width = min(width, max_tokens)
        max_offset = max((max_tokens or width) - width, 0)
        start = (
            _stable_offset(f"{global_step}:{request_id}:window", max_offset)
            if mode == "random"
            else 0
        )
        end = start + width
    else:
        return None, f"unsupported_window_mode:{mode}"
    if start < 0 or end <= start or min_rows <= 0:
        return None, "invalid_window"
    if max_tokens is not None and max_tokens - start < min_rows:
        return None, "generation_budget_too_short"

    method = _speculative_method(engine, drafter_config)
    aux_layer_ids = _resolve_aux_layer_ids(engine)
    needs_aux = method in {"eagle3", "dflash", "dflash2", "domino", "dspark"}
    if needs_aux and not aux_layer_ids:
        return None, "aux_layers_unavailable"
    hidden_layout = (
        resolve_drafter_hidden_states_layout(
            _get(drafter_config, "speculative_algorithm", method), training
        )
        if aux_layer_ids
        else "final"
    )
    try:
        from vllm.v1.hidden_state_capture import HiddenStateCapturePlan

        plan = HiddenStateCapturePlan.from_window(
            request_id=str(request_id),
            prompt_len=len(prompt_ids),
            start=start,
            end=end,
            collection_id=f"vllm:{global_step}:{replica_rank}:{request_id}",
            coordinate=coordinate,
            min_rows=min_rows,
            aux_layer_ids=aux_layer_ids,
            hidden_layout=(
                "aux_final" if hidden_layout.endswith("_plus_last") else hidden_layout
            ),
        )
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        return None, f"plan_unavailable:{type(exc).__name__}"
    return (
        VLLMHiddenCaptureContext(
            request_id=str(request_id),
            plan=plan,
            prompt_ids=list(prompt_ids),
            global_step=int(global_step),
            replica_rank=int(replica_rank),
            window_mode=mode,
            training_hidden_layout=hidden_layout,
        ),
        None,
    )


def vllm_capture_to_drafter_sample(
    context: VLLMHiddenCaptureContext, token_ids: list[int]
) -> dict[str, Any] | None:
    result = context.result
    if result is None:
        return None
    import numpy as np
    import torch

    positions = np.asarray(result.hidden_positions, dtype=np.int64)
    hidden_states = result.hidden_states.detach().cpu()
    if positions.ndim != 1 or len(positions) != int(hidden_states.shape[0]):
        return None
    if len(positions) == 0 or (
        len(positions) > 1 and not np.all(np.diff(positions) == 1)
    ):
        return None
    response_ids = list(token_ids or context.output_token_ids)
    prompt = torch.tensor(context.prompt_ids, dtype=torch.long)
    response = torch.tensor(response_ids, dtype=torch.long)
    return {
        "input_ids": torch.cat([prompt, response]).unsqueeze(0),
        "prompts": prompt.unsqueeze(0),
        "responses": response.unsqueeze(0),
        "hidden_states": hidden_states.unsqueeze(0),
        "hidden_positions": torch.from_numpy(positions.copy()).unsqueeze(0),
        "hidden_position_start": int(positions[0]),
        "hidden_position_end": int(positions[-1]) + 1,
        "hidden_window_mode": context.window_mode,
        "hidden_window_start": int(result.hidden_window_start),
        "hidden_window_end": int(result.hidden_window_end),
        "hidden_window_min_rows": int(context.plan.min_rows),
        "hidden_states_layout": context.training_hidden_layout,
        "global_step": context.global_step,
        "replica_rank": context.replica_rank,
        "collection_id": str(result.collection_id),
    }


def vllm_capture_metrics(
    context: VLLMHiddenCaptureContext | None, skip_reason: str | None
) -> dict[str, Any]:
    result = None if context is None else context.result
    captured_rows = 0 if result is None else len(result.hidden_positions)
    return {
        "drafter/vllm_hidden_capture_armed": int(context is not None),
        "drafter/vllm_hidden_capture_finished": int(result is not None),
        "drafter/vllm_hidden_capture_planned_rows": (
            0
            if context is None
            else context.plan.window_end_abs - context.plan.window_start_abs
        ),
        "drafter/vllm_hidden_capture_rows": captured_rows,
        "drafter/vllm_hidden_capture_copied_bytes": int(
            0 if result is None else result.copied_bytes
        ),
        "drafter/vllm_hidden_capture_copy_ms": float(
            0.0 if result is None else result.copy_ms
        ),
        "drafter/vllm_hidden_capture_skip_reason": (
            skip_reason or (None if context is None else context.skip_reason)
        ),
    }
