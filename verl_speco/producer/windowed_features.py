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

"""Explicit row/layer bridge for windowed vLLM features; legacy payloads stay unchanged."""

from __future__ import annotations
import torch
from verl_speco.trainer.feature_store import DraftFeatureSample


def runner_plan(
    request_id, prompt_len, start, end, auxiliary_output_ids, collection_id
):
    from vllm.v1.hidden_state_capture import HiddenStateCapturePlan

    return HiddenStateCapturePlan.from_window(
        request_id,
        prompt_len,
        start,
        end,
        collection_id,
        min_rows=end - start,
        aux_layer_ids=tuple(auxiliary_output_ids),
        hidden_layout="aux_final",
    )


def connector_window(start, end, *, max_model_len):
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= max_model_len
    ):
        raise ValueError("Window must satisfy 0 <= start < end <= max_model_len")
    return {
        "kv_transfer_params": {
            "hidden_state_window": {"start": start, "end": end, "min_rows": end - start}
        }
    }


def runner_payload(output, prompt_token_ids, generated_token_ids):
    capture = output.hidden_state_capture
    if capture is None:
        raise ValueError(f"Capture unavailable: {output.hidden_capture_skip_reason}")
    if capture.hidden_layout != "aux_final" or not capture.includes_final_layer:
        raise ValueError("DSpark requires auxiliary features and final hidden states")
    positions = torch.as_tensor(capture.hidden_positions, dtype=torch.int64)
    sequence = torch.tensor(
        list(prompt_token_ids) + list(generated_token_ids), dtype=torch.int64
    )
    if positions.numel() == 0 or int(positions[-1]) + 1 >= sequence.numel():
        raise ValueError("Capture has no next-token label")
    n_layers = len(capture.layer_ids) + 1
    hidden = capture.hidden_states
    if hidden.ndim != 2 or hidden.shape[1] % n_layers:
        raise ValueError("Invalid flattened Runner layer layout")
    return {
        "hidden_states": hidden.reshape(len(positions), n_layers, -1),
        "hidden_positions": positions,
        "token_ids": sequence[positions],
        "target_token_ids": sequence[positions + 1],
        "layer_ids": torch.tensor(list(capture.layer_ids) + [-1]),
        "final_is_normalized": torch.tensor(True),
    }


def windowed_feature(payload, request, contract, *, final_norm=None):
    """Convert explicit positions without inferring offsets or applying final norm twice."""
    values = getattr(payload, "payload", payload)
    hidden = values["hidden_states"]
    positions = values["hidden_positions"]
    if (
        not torch.is_tensor(positions)
        or positions.dtype != torch.int64
        or positions.ndim != 1
    ):
        raise ValueError("hidden_positions must be int64 rows")
    positions = positions.cpu()
    if not len(positions) or (positions[1:] <= positions[:-1]).any():
        raise ValueError("Positions must be nonempty, unique and strictly increasing")
    if (
        hidden.ndim != 3
        or hidden.shape[0] != len(positions)
        or not torch.isfinite(hidden).all()
    ):
        raise ValueError("Invalid hidden-state shape or nonfinite features")
    expected = request.feature_positions.cpu().long()
    if not torch.equal(positions, expected):
        raise ValueError(
            "Window does not cover requested features; prompt context cannot be discarded implicitly"
        )
    if int(positions[0]) < 0 or int(positions[-1]) >= request.input_ids.numel():
        raise ValueError("Positions outside request input")
    source_ids = request.input_ids.cpu().long().index_select(0, positions)
    if not torch.equal(values["token_ids"].cpu().long(), source_ids):
        raise ValueError("Window tokens do not belong to this request")
    target_ids = values.get("target_token_ids")
    available = positions + 1 < request.input_ids.numel()
    if target_ids is not None and available.any():
        wanted = request.input_ids.cpu().long()[positions[available] + 1]
        if not torch.equal(target_ids.cpu().long()[available], wanted):
            raise ValueError("Window next-token labels are misaligned")
    auxiliary_ids = contract.vllm_aux_hidden_state_layer_ids
    if auxiliary_ids is None:
        auxiliary_ids = contract.target_layer_ids
    expected_ids = list(auxiliary_ids) + [-1]
    actual_ids = values["layer_ids"].cpu().long().tolist()
    if actual_ids != expected_ids or hidden.shape[1] != len(expected_ids):
        raise ValueError(f"Layer contract mismatch: {actual_ids} != {expected_ids}")
    if (
        contract.algorithm.upper() != "DSPARK"
        or contract.hidden_states_layout != "dflash_aux_plus_last"
    ):
        raise ValueError(
            "This explicit window adapter supports DSpark aux-plus-last only"
        )
    normalized = values.get("final_is_normalized")
    if normalized is None or not torch.is_tensor(normalized) or normalized.numel() != 1:
        raise ValueError("Explicit final normalization metadata is required")
    selected = hidden.to(dtype=contract.dtype)
    final = selected[:, -1, :]
    if not bool(normalized.item()):
        if final_norm is None:
            raise ValueError("Unnormalized final features require target final norm")
        with torch.no_grad():
            final = final_norm(final)
    output = torch.cat((selected[:, :-1].flatten(1), final), dim=-1).cpu().contiguous()
    metadata = dict(
        getattr(request, "source_metadata", getattr(request, "metadata", {})) or {}
    )
    if request.draft_position_ids.numel() != len(positions):
        raise ValueError("Draft position IDs must cover every feature row")
    metadata.update(
        source=contract.source,
        hidden_states_layout=contract.hidden_states_layout,
        target_layer_ids=list(contract.target_layer_ids),
        vllm_aux_hidden_state_layer_ids=list(auxiliary_ids),
        hidden_positions=positions,
        hidden_position_start=int(positions[0]),
        hidden_position_end=int(positions[-1]) + 1,
        feature_start=int(positions[0]),
        feature_end=int(positions[-1]) + 1,
        sequence_length=len(positions),
        full_sequence_length=request.input_ids.numel(),
        last_hidden_state_norm="target_final_norm",
    )
    return DraftFeatureSample(
        algorithm="DSPARK",
        input_ids=source_ids,
        loss_mask=request.loss_mask.cpu().index_select(0, positions).float(),
        hidden_states=output,
        position_ids=request.draft_position_ids.cpu().long(),
        metadata=metadata,
    )


def mooncake_safe_payload(payload):
    """Encode scalar metadata as one-element tensors for hs-connectors 0.8."""
    return {
        name: value.reshape(1) if value.ndim == 0 else value.contiguous()
        for name, value in payload.items()
    }
