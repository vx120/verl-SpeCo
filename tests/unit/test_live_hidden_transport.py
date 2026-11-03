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

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def transport(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "live",
        Path(__file__).resolve().parents[2]
        / "verl_speco/integration/live_hidden_transport.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stored, cleared = {}, []

    def clear(keys):
        cleared.extend(keys)
        for key in keys:
            stored.pop(key, None)

    bridge = types.SimpleNamespace(
        configure_transfer_queue=lambda cfg: True,
        put_sample=lambda key, fields, tag: stored.update({key: fields}),
        get_samples=lambda keys: [(key, stored[key]) for key in keys],
        clear_samples=clear,
        list_samples=lambda: list(stored),
    )
    integration = types.ModuleType("verl_speco.integration")
    integration.transferqueue_bridge = bridge
    monkeypatch.setitem(sys.modules, "verl_speco", types.ModuleType("verl_speco"))
    monkeypatch.setitem(sys.modules, "verl_speco.integration", integration)
    return module, stored, cleared, bridge


def config():
    return {
        "transfer_queue": {
            "enable": True,
            "run_id": "test",
            "backend": {"storage_backend": "MooncakeStore"},
        }
    }


def sample():
    return {
        "hidden_states": torch.arange(12).reshape(1, 3, 4).to(torch.bfloat16),
        "global_step": 2,
        "replica_rank": 1,
        "collection_id": "request:abc",
        "hidden_positions": torch.tensor([[11, 12, 13]]),
    }


def test_reference_roundtrip_and_release_after_multiple_readers(transport):
    module, stored, cleared, _ = transport
    original = sample()
    ref = module.put_hidden(original, config())
    assert "hidden_states" not in ref
    assert ref["replica_rank"] == 1 and ref["global_step"] == 2
    assert torch.equal(ref["hidden_positions"], original["hidden_positions"])
    for _ in range(2):
        assert torch.equal(module.get_hidden(ref, config()), original["hidden_states"])
        assert not cleared
    keys = list(stored)
    module.release_collection(types.SimpleNamespace(buckets=[[ref], [ref]]))
    assert cleared == keys
    assert not stored


def test_corruption_is_rejected(transport):
    module, stored, _, _ = transport
    ref = module.put_hidden(sample(), config())
    stored[ref["hidden_states_tq_ref"]["key"]]["hidden_states"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="checksum"):
        module.get_hidden(ref, config())


def test_unavailable_transport_fails_without_inline_fallback(transport):
    module, _, _, bridge = transport
    bridge.configure_transfer_queue = lambda cfg: False
    with pytest.raises(RuntimeError, match="unavailable"):
        module.put_hidden(sample(), config())


def test_disabled_preserves_existing_sample(transport):
    module, stored, _, _ = transport
    original = sample()
    assert module.put_hidden(original, {}) is original
    assert module.get_hidden(original, {}) is original["hidden_states"]
    assert not stored


def tensor_sample():
    return {
        "global_step": 2,
        "replica_rank": 0,
        "collection_id": "teacher",
        "logits_contract": {"vocab_size": 100},
        "tensors": {
            "hidden_states": torch.arange(28).reshape(7, 4).to(torch.bfloat16),
            "hidden_positions": torch.arange(3, 10),
            "response_token_ids": torch.arange(50, 57),
        },
    }


def test_tensor_chunks_preserve_all_fields_for_multiple_readers(transport):
    module, stored, cleared, _ = transport
    original = tensor_sample()
    ref = module.put_tensor_chunks(original, config(), chunk_rows=3)
    assert "tensors" not in ref
    assert ref["logits_contract"] == original["logits_contract"]
    assert [(r["start"], r["end"]) for r in ref["tensor_chunks"]] == [
        (0, 3),
        (3, 6),
        (6, 7),
    ]
    assert len(stored) == 3
    for _ in range(2):
        chunks = list(module.iter_tensor_chunks(ref, config()))
        for name, expected in original["tensors"].items():
            assert torch.equal(torch.cat([chunk[name] for chunk in chunks]), expected)
        assert not cleared
    module.release_tensor_chunks(ref)
    assert not stored and len(cleared) == 3


def test_partial_tensor_publication_releases_attempted_chunks(transport):
    module, stored, _, bridge = transport

    def fail_second(key, fields, tag):
        stored[key] = fields
        if len(stored) == 2:
            raise RuntimeError("publication failed")

    bridge.put_sample = fail_second
    with pytest.raises(RuntimeError, match="publication failed"):
        module.put_tensor_chunks(tensor_sample(), config(), chunk_rows=3)
    assert not stored


def test_tensor_chunk_corruption_fails_before_delivery(transport):
    module, stored, _, _ = transport
    ref = module.put_tensor_chunks(tensor_sample(), config(), chunk_rows=3)
    stored[ref["tensor_chunks"][0]["key"]]["response_token_ids"][0] += 1
    try:
        with pytest.raises(ValueError, match="integrity"):
            list(module.iter_tensor_chunks(ref, config()))
    finally:
        module.release_tensor_chunks(ref)
    assert not stored


def test_incomplete_tensor_manifest_is_rejected(transport):
    module, stored, _, _ = transport
    ref = module.put_tensor_chunks(tensor_sample(), config(), chunk_rows=3)
    incomplete = {**ref, "tensor_chunks": ref["tensor_chunks"][:-1]}
    with pytest.raises(ValueError, match="incomplete"):
        list(module.iter_tensor_chunks(incomplete, config()))
    module.release_tensor_chunks(ref)
    assert not stored


def test_tensor_chunks_disabled_keep_inline_payload(transport):
    module, stored, _, _ = transport
    sample = tensor_sample()
    assert module.put_tensor_chunks(sample, {}) is sample
    assert next(module.iter_tensor_chunks(sample, {})) is sample["tensors"]
    assert not stored


def test_default_tensor_delivery_avoids_small_rpc_chunks(transport):
    module, stored, _, _ = transport
    original = tensor_sample()
    original["tensors"] = {
        "hidden_states": torch.ones(2048, 2048, dtype=torch.bfloat16),
        "hidden_positions": torch.arange(2048),
        "response_token_ids": torch.arange(2048),
    }
    ref = module.put_tensor_chunks(original, config())
    assert len(ref["tensor_chunks"]) == 1
    assert len(stored) == 1
    assert torch.equal(
        next(module.iter_tensor_chunks(ref, config()))["hidden_states"],
        original["tensors"]["hidden_states"],
    )
    module.release_tensor_chunks(ref)


def test_default_tensor_delivery_bounds_large_payload_chunks(transport):
    module, _, _, _ = transport
    original = tensor_sample()
    original["tensors"] = {
        "hidden_states": torch.ones(4097, 2048, dtype=torch.bfloat16)
    }
    ref = module.put_tensor_chunks(original, config())
    assert [(c["start"], c["end"]) for c in ref["tensor_chunks"]] == [
        (0, 4096),
        (4096, 4097),
    ]
    module.release_tensor_chunks(ref)
