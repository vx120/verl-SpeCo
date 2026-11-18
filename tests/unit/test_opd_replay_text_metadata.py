"""Text agent-loop metadata must not be mistaken for real multimodal input."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl_speco.integration.opd_replay import expand_replay_batch
from verl_speco.integration.verify_trace import TRACE_KEY


def make_source():
    trace = {"layout": "dspark", "num_spec_tokens": 3, "prompt_length": 3,
             "response_length": 2, "steps": [{"anchor": 2, "commit": 2,
             "drafts": 3, "accepted": 1, "raw_generated": 2,
             "truncated": False, "draft_token_ids": [4, 5, 8]}]}
    ids = torch.tensor([[1, 2, 3, 4, 9], [1, 2, 3, 4, 9]])
    batch = SimpleNamespace(batch={"input_ids": ids, "prompts": ids[:, :3],
        "responses": ids[:, 3:], "attention_mask": torch.ones_like(ids),
        "position_ids": torch.arange(5).expand(2, -1)},
        non_tensor_batch={TRACE_KEY: [trace, None]})
    plan = {"collect_mask": torch.tensor([True, False]),
        "owner_rank": torch.tensor([0, -1]), "hidden_positions": torch.zeros(2, 8, dtype=torch.long)}
    return batch, plan


@pytest.mark.parametrize("metadata", [None, {}, [], [None, None], [{}, {}],
    [None, {}], np.array([{}, {}], dtype=object), np.array([None, {}], dtype=object)])
def test_text_only_empty_multimodal_payloads_are_equivalent(metadata):
    batch, plan = make_source()
    expected = expand_replay_batch(batch, plan, max_blocks=1)
    batch.non_tensor_batch["multi_modal_inputs"] = copy.deepcopy(metadata)
    result = expand_replay_batch(batch, plan, max_blocks=1)
    for key in expected[0]:
        torch.testing.assert_close(expected[0][key], result[0][key])
    assert result[1][TRACE_KEY][2]["draft_token_ids"] == [4, 5, 8]
    assert result[0]["input_ids"][2, result[0]["attention_mask"][2].bool()].tolist() == [1, 2, 3, 4, 5, 8]
    assert "multi_modal_inputs" in batch.non_tensor_batch


@pytest.mark.parametrize("metadata", [[{"pixel_values": torch.ones(1)}, {}],
    {"image_grid_thw": torch.ones(1)}, [None, {"video": []}], ["unexpected", None]])
def test_actual_multimodal_payloads_are_rejected(metadata):
    batch, plan = make_source()
    batch.non_tensor_batch["multi_modal_inputs"] = metadata
    with pytest.raises(ValueError, match="nonempty multimodal inputs"):
        expand_replay_batch(batch, plan, max_blocks=1)


def test_nontext_position_shape_has_separate_diagnostic():
    batch, plan = make_source()
    batch.non_tensor_batch["multi_modal_inputs"] = [{}, {}]
    batch.batch["position_ids"] = torch.zeros(2, 3, 5, dtype=torch.long)
    with pytest.raises(ValueError, match=r"shape=\(2, 3, 5\)"):
        expand_replay_batch(batch, plan, max_blocks=1)


def test_real_upstream_text_agent_loop_emits_empty_dict():
    import ast
    source = Path(__file__).resolve().parents[3] / "Draft-OPD/verl/verl/experimental/agent_loop/agent_loop.py"
    if not source.exists():
        pytest.skip("Upstream source not present")
    fn = next(n for n in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
              if isinstance(n, ast.FunctionDef) and n.name == "_compute_multi_modal_inputs")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    payload = ns[fn.name](SimpleNamespace(processor=None), None, None)
    assert payload == {}
    batch, plan = make_source()
    batch.non_tensor_batch["multi_modal_inputs"] = np.array([payload, payload], dtype=object)
    assert expand_replay_batch(batch, plan, max_blocks=1) is not None
