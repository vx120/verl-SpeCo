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

from types import SimpleNamespace as NS
import pytest
import torch
from verl_speco.producer.windowed_features import windowed_feature, connector_window
from verl_speco.trainer.target_feature_replay import feature_from_vllm_payload


def fixture():
    request = NS(
        input_ids=torch.arange(20),
        feature_positions=torch.arange(5, 13),
        loss_mask=torch.ones(20),
        draft_position_ids=torch.arange(6, 14),
        source_metadata={},
    )
    contract = NS(
        algorithm="DSPARK",
        vllm_aux_hidden_state_layer_ids=[2, 10, 18, 26, 34],
        target_layer_ids=[1, 9, 17, 25, 33],
        hidden_states_layout="dflash_aux_plus_last",
        dtype=torch.float32,
        source="test",
    )
    payload = {
        "token_ids": torch.arange(5, 13),
        "hidden_positions": torch.arange(5, 13),
        "hidden_states": torch.arange(8 * 6 * 4).reshape(8, 6, 4).float(),
        "layer_ids": torch.tensor([2, 10, 18, 26, 34, -1]),
        "target_token_ids": torch.arange(6, 14),
        "final_is_normalized": torch.tensor(True),
    }
    return payload, request, contract


def test_explicit_window_not_suffix_and_norm_applied_once():
    p, r, c = fixture()

    def norm(x):
        raise AssertionError("normalized twice")

    actual = feature_from_vllm_payload(p, r, c, final_norm=norm)
    torch.testing.assert_close(actual.hidden_states, p["hidden_states"].flatten(1))
    torch.testing.assert_close(actual.input_ids, torch.arange(5, 13))
    assert actual.metadata["hidden_position_start"] == 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("token_ids", torch.arange(6, 14)),
        ("target_token_ids", torch.arange(7, 15)),
        ("hidden_positions", torch.tensor([5, 6, 7, 8, 9, 10, 11, 11])),
        ("layer_ids", torch.tensor([1, 9, 17, 25, 33, -1])),
    ],
)
def test_cross_request_position_and_layer_corruption_rejected(field, value):
    p, r, c = fixture()
    p[field] = value
    with pytest.raises(ValueError):
        windowed_feature(p, r, c)


def test_prompt_context_requirement_not_silently_removed():
    p, r, c = fixture()
    r.feature_positions = torch.arange(13)
    with pytest.raises(ValueError, match="prompt context"):
        windowed_feature(p, r, c)


def test_raw_final_norm_and_nonfinite_validation():
    p, r, c = fixture()
    p["final_is_normalized"] = torch.tensor(False)
    result = windowed_feature(p, r, c, final_norm=lambda x: x * 2)
    torch.testing.assert_close(
        result.hidden_states[:, -4:], p["hidden_states"][:, -1, :] * 2
    )
    p["hidden_states"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        windowed_feature(p, r, c, final_norm=lambda x: x)


def test_window_int_overflow_rejected_before_engine():
    with pytest.raises(ValueError):
        connector_window(0, 2**63, max_model_len=512)


def test_failed_export_marker_prevents_loading_stale_payload(tmp_path):
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore

    path = tmp_path / "features.safetensors"
    (tmp_path / "features.safetensors.error").write_text("disk write failed")
    store = FileHiddenStatesStore()
    with pytest.raises(RuntimeError, match="disk write failed"):
        store.load(str(path))
    store.release(str(path))
    assert not (tmp_path / "features.safetensors.error").exists()


def test_mooncake_scalar_wire_metadata_preserves_value():
    from verl_speco.producer.windowed_features import mooncake_safe_payload

    wire = mooncake_safe_payload(
        {"final_is_normalized": torch.tensor(True), "hidden_positions": torch.arange(3)}
    )
    assert wire["final_is_normalized"].shape == (1,)
    assert wire["final_is_normalized"].item() is True
    torch.testing.assert_close(wire["hidden_positions"], torch.arange(3))
