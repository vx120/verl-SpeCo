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

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "connector_adapter",
    Path(__file__).resolve().parents[2] / "verl_speco/integration/connector_cotrain.py",
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def payload():
    return {
        "hidden_positions": torch.tensor([2, 3, 4]),
        "hidden_states": torch.arange(36).reshape(3, 3, 4).float(),
        "token_ids": torch.tensor([3, 4, 5]),
        "target_token_ids": torch.tensor([4, 5, 6]),
        "layer_ids": torch.tensor([2, 10, -1]),
    }


def convert(value):
    return adapter.payload_to_sample(
        value,
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5, 6],
        step=2,
        owner=1,
        collection_id="request",
        layers=[2, 10],
    )


def test_actual_rollout_positions_and_layer_flattening():
    original = payload()
    result = convert(original)
    assert result["hidden_states"].shape == (1, 3, 12)
    assert torch.equal(result["hidden_states"][0], original["hidden_states"].flatten(1))
    assert result["hidden_position_start"] == 2 and result["hidden_position_end"] == 5
    assert result["replica_rank"] == 1 and result["global_step"] == 2
    assert result["hidden_states_layout"] == "dflash_aux_plus_last"


@pytest.mark.parametrize(
    "key", ["hidden_positions", "token_ids", "target_token_ids", "layer_ids"]
)
def test_mismatched_metadata_rejected(key):
    value = payload()
    value[key][0] += 1
    with pytest.raises(ValueError):
        convert(value)


def test_nonfinite_rejected():
    value = payload()
    value["hidden_states"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Invalid Connector"):
        convert(value)


def test_engine_is_extraction_with_explicit_layout():
    settings = adapter.engine_settings(
        {
            "connector_export_path": "/tmp/isolated",
            "connector_target_layer_ids": [2, 10],
        }
    )
    assert settings["speculative_config"]["method"] == "extract_hidden_states"
    assert (
        settings["speculative_config"]["draft_model_config"]["hf_config"][
            "hidden_states_layout"
        ]
        == "auxiliary_plus_final"
    )
    assert settings["kv_transfer_config"]["kv_connector_extra_config"][
        "use_synchronization_lock"
    ]
