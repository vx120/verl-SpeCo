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
from pathlib import Path
import sys
import types

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
