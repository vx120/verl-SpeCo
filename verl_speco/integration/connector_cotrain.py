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

"""Connector online training adapter; extraction occupies the drafter method.

This variant trains DSpark from RL rollout features but does not serve DSpark
speculation or publish its weights into the extraction engine.
"""

import asyncio
from contextvars import ContextVar
import inspect
import logging
import os
from pathlib import Path

_ACTIVE = ContextVar("connector_cotrain_request", default=None)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def engine_settings(training):
    root = training["connector_export_path"]
    layers = list(training["connector_target_layer_ids"])
    return {
        "speculative_config": {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": layers,
                    "hidden_states_layout": "auxiliary_plus_final",
                }
            },
        },
        "kv_transfer_config": {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": root,
                "enable_window_capture": True,
                "use_synchronization_lock": True,
            },
        },
    }


def install(engine):
    if getattr(engine, "_speco_connector_wrapped", False):
        return
    original = engine.generate

    async def generate(*args, **kwargs):
        active = _ACTIVE.get()
        if active is not None:
            params = kwargs["sampling_params"]
            extra = dict(params.extra_args or {})
            transfer = dict(extra.get("kv_transfer_params") or {})
            transfer["hidden_state_window"] = {
                "start": 0,
                "end": active["rows"],
                "min_rows": active["min_rows"],
            }
            extra["kv_transfer_params"] = transfer
            params.extra_args = extra
        iterator = original(*args, **kwargs)
        if inspect.isawaitable(iterator):
            iterator = await iterator
        async for item in iterator:
            if active is not None:
                transfer = getattr(item, "kv_transfer_params", None) or {}
                if transfer.get("hidden_states_path"):
                    active["path"] = transfer["hidden_states_path"]
            yield item

    engine.generate = generate
    engine._speco_connector_wrapped = True


def payload_to_sample(
    payload, *, prompt_ids, response_ids, step, owner, collection_id, layers
):
    import torch

    positions = payload["hidden_positions"].long()
    hidden = payload["hidden_states"]
    rows = len(positions)
    expected = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + rows)
    if not torch.equal(positions, expected):
        raise ValueError(
            "Connector hidden positions differ from rollout response-source positions"
        )
    sequence = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
    if not rows or int(positions[-1]) + 1 >= len(sequence):
        raise ValueError("Connector window has no next-token supervision")
    if not torch.equal(payload["token_ids"], sequence[positions]):
        raise ValueError("Connector exported token IDs differ from rollout")
    if not torch.equal(payload["target_token_ids"], sequence[positions + 1]):
        raise ValueError("Connector exported labels differ from rollout")
    if payload["layer_ids"].tolist() != layers + [-1]:
        raise ValueError("Connector auxiliary/final layer order differs from trainer")
    if (
        hidden.ndim != 3
        or hidden.shape[1] != len(layers) + 1
        or not torch.isfinite(hidden).all()
    ):
        raise ValueError("Invalid Connector hidden states")
    return {
        "input_ids": sequence.unsqueeze(0),
        "prompts": torch.tensor(prompt_ids).unsqueeze(0),
        "responses": torch.tensor(response_ids).unsqueeze(0),
        "hidden_states": hidden.flatten(1).unsqueeze(0),
        "hidden_positions": positions.unsqueeze(0),
        "hidden_position_start": int(positions[0]),
        "hidden_position_end": int(positions[-1]) + 1,
        "hidden_window_mode": "prefix",
        "hidden_window_start": 0,
        "hidden_window_end": rows,
        "hidden_states_layout": "dflash_aux_plus_last",
        "global_step": step,
        "replica_rank": owner,
        "collection_id": collection_id,
    }


def read_export(path, root):
    import fcntl
    from safetensors.torch import load_file

    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(
            "Connector export is outside the configured experiment directory"
        )
    with open(str(resolved) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        error = Path(str(resolved) + ".error")
        if error.exists():
            raise RuntimeError(error.read_text())
        return load_file(str(resolved))


async def generate_sample(server, base_generate, drafter, *args, **kwargs):
    training = drafter["training"]
    prompt = list(kwargs.get("prompt_ids", args[0] if args else []))
    request_id = str(kwargs.get("request_id", args[2] if len(args) > 2 else ""))
    step = int(getattr(server, "_speco_capture_global_step", 0))
    active = {
        "rows": int(training.get("dspark_max_window", 32)),
        "min_rows": int(training.get("hidden_state_window_min_rows", 8)),
    }
    install(server.engine)
    token = _ACTIVE.set(active if step > 0 else None)
    try:
        output = await base_generate(*args, **kwargs)
    finally:
        _ACTIVE.reset(token)
    if step <= 0:
        return output
    if "path" not in active:
        raise RuntimeError(f"Connector did not return an export path for {request_id}")
    payload = await asyncio.to_thread(
        read_export, active["path"], training["connector_export_path"]
    )
    sample = payload_to_sample(
        payload,
        prompt_ids=prompt,
        response_ids=list(output.token_ids),
        step=step,
        owner=int(getattr(server, "replica_rank", 0)),
        collection_id=f"connector:{step}:{request_id}",
        layers=list(training["connector_target_layer_ids"]),
    )
    from verl_speco.integration.live_hidden_transport import put_hidden

    output.extra_fields["drafter_sample"] = put_hidden(sample, training)
    output.extra_fields["drafter/connector_capture_rows"] = int(
        payload["hidden_states"].shape[0]
    )
    logger.info(
        "[connector cotrain capture] step=%s owner=%s rows=%s positions=[%s,%s) file=%s",
        step,
        sample["replica_rank"],
        len(payload["hidden_positions"]),
        sample["hidden_position_start"],
        sample["hidden_position_end"],
        active["path"],
    )
    return output
