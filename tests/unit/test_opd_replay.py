import copy
from types import SimpleNamespace

import pytest
import torch

from verl_speco.backends.drafter_opd import DraftOPDObjective, preprocess_trace
from verl_speco.integration.opd_replay import expand_replay_batch
from verl_speco.integration.verify_trace import (
    TRACE_KEY,
    finish_verify_step,
    snapshot_verify_step,
    validate_verify_trace,
)


def source(layout="dspark"):
    trace = {
        "layout": layout,
        "num_spec_tokens": 3,
        "prompt_length": 3,
        "response_length": 4,
        "steps": [
            {
                "anchor": 2,
                "commit": 2,
                "drafts": 3,
                "accepted": 1,
                "raw_generated": 2,
                "truncated": False,
                "draft_token_ids": [4, 5, 8],
            },
            {
                "anchor": 4,
                "commit": 1,
                "drafts": 0,
                "accepted": 0,
                "raw_generated": 1,
                "truncated": False,
                "draft_token_ids": [],
            },
            {
                "anchor": 5,
                "commit": 1,
                "drafts": 0,
                "accepted": 0,
                "raw_generated": 1,
                "truncated": False,
                "draft_token_ids": [],
            },
        ],
    }
    ids = torch.tensor([[1, 2, 3, 4, 9, 6, 7], [1, 2, 3, 4, 9, 6, 7]])
    batch = SimpleNamespace(
        batch={
            "input_ids": ids,
            "prompts": ids[:, :3],
            "responses": ids[:, 3:],
            "attention_mask": torch.ones_like(ids),
            "position_ids": torch.arange(7).expand(2, -1),
            "response_mask": torch.ones(2, 4, dtype=torch.long),
        },
        non_tensor_batch={TRACE_KEY: [trace, None]},
    )
    plan = {
        "collect_mask": torch.tensor([True, False]),
        "owner_rank": torch.tensor([0, -1]),
        "hidden_positions": torch.zeros(2, 8, dtype=torch.long),
    }
    return batch, plan


def test_scheduler_keeps_proposals_before_correction_and_request_removal():
    request = SimpleNamespace(
        prompt_token_ids=[1, 2, 3], _output_token_ids=[], is_finished=lambda: False
    )
    scheduler = SimpleNamespace(requests={"r": request})
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"r": 4},
        scheduled_spec_decode_tokens={"r": [4, 5, 8, -1]},
        num_invalid_spec_tokens={"r": 1},
    )
    output = SimpleNamespace(req_id_to_index={"r": 0}, sampled_token_ids=[[4, 9]])
    snapshots = snapshot_verify_step(
        scheduler, scheduled, output, capture_draft_tokens=True
    )
    scheduled.scheduled_spec_decode_tokens["r"].clear()
    request._output_token_ids.extend([4, 9])
    scheduler.requests.clear()
    trace = finish_verify_step(snapshots, layout="dspark", capacity=3)["r"]
    assert trace["steps"][0]["draft_token_ids"] == [4, 5, 8]
    validate_verify_trace(trace, layout="dspark", capacity=3)


@pytest.mark.parametrize("layout", ["dflash", "dspark"])
def test_expansion_keeps_policy_rows_and_scores_full_proposal_branch(layout):
    batch, plan = source(layout)
    original = batch.batch["input_ids"].clone()
    tensors, metadata, capture = expand_replay_batch(batch, plan, max_blocks=0)
    assert tensors["input_ids"].shape[0] == 4  # padded to original dispatch multiple
    assert capture["collect_mask"].tolist() == [False, False, True, False]
    for row in range(2):
        assert (
            tensors["input_ids"][row, tensors["attention_mask"][row].bool()].tolist()
            == original[row].tolist()
        )
    branch = tensors["input_ids"][2, tensors["attention_mask"][2].bool()]
    assert branch.tolist() == [1, 2, 3, 4, 5, 8]
    assert metadata[TRACE_KEY][2]["accepted"] == 1
    assert tensors["response_mask"][2, :3].tolist() == [1, 1, 1]
    assert capture["hidden_positions"][
        2, capture["hidden_position_mask"][2]
    ].tolist() == list(range(6))
    full_plan = preprocess_trace(
        {TRACE_KEY: metadata[TRACE_KEY][2]},
        start=0,
        end=6,
        layout=layout,
        capacity=3,
        device="cpu",
        require_full_suffix=True,
    )
    assert full_plan[2].tolist() == [1, 3, 1]
    with pytest.raises(ValueError, match="actor-scored"):
        preprocess_trace(
            {TRACE_KEY: batch.non_tensor_batch[TRACE_KEY][0]},
            start=0,
            end=7,
            layout=layout,
            capacity=3,
            device="cpu",
            require_full_suffix=True,
        )
    torch.testing.assert_close(batch.batch["input_ids"], original)


def test_teacher_recomputes_rejected_prefix_and_suffix_has_draft_gradient():
    torch.manual_seed(91)
    batch, plan = source()
    tensors, metadata, _ = expand_replay_batch(batch, plan, max_blocks=0)
    embedding = torch.nn.Embedding(16, 4)
    teacher = torch.nn.GRU(4, 4, batch_first=True)
    head = torch.randn(16, 4)
    with torch.no_grad():
        branch_ids = tensors["input_ids"][2, tensors["attention_mask"][2].bool()]
        branch_hidden = teacher(embedding(branch_ids).unsqueeze(0))[0][0]
        corrected_hidden = teacher(embedding(batch.batch["input_ids"][:1]))[0][0]
    # Position 4 predicts the third proposal; its predecessor must be draft 5,
    # whereas the corrected continuation contains 9 at the same position.
    assert not torch.allclose(branch_hidden[4], corrected_hidden[4])
    logits = torch.randn(3, 16, requires_grad=True)
    objective = DraftOPDObjective(
        {"replay_full_suffix": True},
        layout="dspark",
        loss_mode="full_vocab",
        per_position_loss_weight="fixed_exp_decay",
    )
    mask = torch.ones(1, 1, 3)
    loss = objective(
        logits,
        branch_hidden[2:5],
        head,
        mask,
        mask.flatten().bool(),
        position_weights=mask,
    )
    tv = (logits.softmax(-1) - (branch_hidden[2:5] @ head.T).softmax(-1)).abs().sum(
        -1
    ) / 2
    expected = 1 - (1 - tv).cumprod(0).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
    )
    loss.backward()
    assert logits.grad[-1].abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_missing_proposals_and_short_capture_fail_closed():
    batch, plan = source()
    del batch.non_tensor_batch[TRACE_KEY][0]["steps"][0]["draft_token_ids"]
    with pytest.raises(ValueError, match="missing proposal"):
        expand_replay_batch(batch, plan, max_blocks=0)
    batch, plan = source()
    plan["hidden_positions"] = torch.zeros(2, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="capture window"):
        expand_replay_batch(batch, plan, max_blocks=0)


@pytest.mark.parametrize("changes", [{"anchor_planner": "mixed"}, {"loss": "ce"}])
def test_full_replay_rejects_objectives_that_would_imitate_rejected_tokens(changes):
    with pytest.raises(ValueError, match="full replay requires"):
        DraftOPDObjective(
            {"replay_full_suffix": True, **changes},
            layout="dspark",
            loss_mode="full_vocab",
            per_position_loss_weight="fixed_exp_decay",
        )


def test_stop_truncated_blocks_are_excluded():
    batch, plan = source()
    trace = batch.non_tensor_batch[TRACE_KEY][0]
    trace["steps"] = [copy.deepcopy(trace["steps"][0])]
    trace["steps"][0].update(commit=1, truncated=True)
    trace["response_length"] = 1
    assert expand_replay_batch(batch, plan, max_blocks=0) is None


def test_all_accepted_block_excludes_bonus_and_window_crop_keeps_suffix():
    batch, plan = source()
    trace = batch.non_tensor_batch[TRACE_KEY][0]
    trace["steps"] = [trace["steps"][0]]
    trace["steps"][0].update(accepted=3, commit=4, raw_generated=4)
    batch.batch["input_ids"][0, 3:] = torch.tensor([4, 5, 8, 9])
    tensors, metadata, _ = expand_replay_batch(batch, plan, max_blocks=0)
    branch = tensors["input_ids"][2, tensors["attention_mask"][2].bool()]
    assert branch.tolist() == [1, 2, 3, 4, 5, 8]
    item = {TRACE_KEY: metadata[TRACE_KEY][2], "_verl_feature_start": 1}
    cropped = preprocess_trace(
        item,
        start=0,
        end=5,
        layout="dspark",
        capacity=3,
        device="cpu",
        require_full_suffix=True,
    )
    assert cropped[1].tolist() == [3, 3, 0]
    with pytest.raises(ValueError, match="cuts the proposal"):
        preprocess_trace(
            item,
            start=0,
            end=4,
            layout="dspark",
            capacity=3,
            device="cpu",
            require_full_suffix=True,
        )


def test_block_budget_samples_whole_proposals_reproducibly():
    batch, plan = source()
    ids = torch.tensor([[1, 2, 3, 4, 9, 6, 9, 8, 9]]).expand(2, -1)
    batch.batch.update(
        input_ids=ids,
        prompts=ids[:, :3],
        responses=ids[:, 3:],
        attention_mask=torch.ones_like(ids),
        position_ids=torch.arange(9).expand(2, -1),
        response_mask=torch.ones(2, 6, dtype=torch.long),
    )
    trace = batch.non_tensor_batch[TRACE_KEY][0]
    trace["response_length"] = 6
    template = trace["steps"][0]
    trace["steps"] = [
        {**template, "anchor": anchor, "draft_token_ids": proposals}
        for anchor, proposals in [(2, [4, 10, 11]), (4, [6, 12, 13]), (6, [8, 14, 15])]
    ]
    _, first, sampled = expand_replay_batch(batch, plan, max_blocks=2, seed=77)
    _, second, _ = expand_replay_batch(batch, plan, max_blocks=2, seed=77)
    _, all_blocks, all_plan = expand_replay_batch(batch, plan, max_blocks=0)
    assert sampled["selected_count"] == 2 and all_plan["selected_count"] == 3
    assert first[TRACE_KEY].tolist() == second[TRACE_KEY].tolist()
    assert {
        trace["anchor"]
        for trace in all_blocks[TRACE_KEY]
        if trace and trace.get("kind")
    } == {2, 4, 6}


@pytest.mark.parametrize("short_response", [False, True])
def test_production_oldlogprob_hook_strips_replay_rows_before_ppo(
    short_response, monkeypatch
):
    """Run the real nested hook with real DataProto and packed-padding helpers.

    Only the actor RPC and feature transport are replaced; the trainer's optional
    distributed dependency stack is not needed to execute this function body.
    """
    import ast
    from pathlib import Path
    import time
    import uuid
    from typing import Any, cast

    from verl import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
    from verl.workers.utils import padding
    from verl_speco.integration import oldlogprob_runtime
    from verl_speco.trainer.scheduler import DrafterCollectionSource

    # The production padding adapter normally delegates these two primitives
    # to flash-attn. Use equivalent torch indexing for the CPU-only test.
    def cpu_unpad(values, mask):
        indices = mask.flatten().nonzero().flatten()
        lengths = mask.sum(-1).int()
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).int()))
        return values.flatten(0, 1)[indices], indices, offsets, int(lengths.max())

    monkeypatch.setattr(padding, "unpad_input", cpu_unpad)
    monkeypatch.setattr(
        padding, "index_first_axis", lambda values, indices: values[indices]
    )

    batch, plan = source()
    if short_response:
        trace = batch.non_tensor_batch[TRACE_KEY][0]
        trace["steps"] = [trace["steps"][0]]
        trace["steps"][0].update(accepted=0, commit=1, raw_generated=1)
        trace["response_length"] = 1
        for key in ("input_ids", "attention_mask", "position_ids"):
            batch.batch[key] = batch.batch[key][:, :4]
        batch.batch["input_ids"][:, 3] = 9
        batch.batch["responses"] = batch.batch["input_ids"][:, 3:]
        batch.batch["response_mask"] = torch.ones(2, 1, dtype=torch.long)
    import numpy as np

    original = DataProto.from_dict(
        tensors=batch.batch,
        non_tensors={
            TRACE_KEY: np.asarray(batch.non_tensor_batch[TRACE_KEY], dtype=object)
        },
    )
    actor_calls, collected, dispatched = [], [], []
    plan.update(
        owner_count=1, collection_plan=SimpleNamespace(collection_id="replay-test")
    )

    def actor_forward(data):
        actor_calls.append(data)
        from tensordict import NonTensorData

        return {
            "log_probs": data["input_ids"].values().float(),
            "metrics": NonTensorData({"mfu": 0.5}),
            oldlogprob_runtime.OLD_LOGPROB_HIDDEN_STATES_KEY: torch.arange(4 * 8 * 12)
            .reshape(4, 8, 12)
            .float(),
        }

    def collect(scored_batch, capture_plan, output):
        assert len(scored_batch) == 4
        assert capture_plan["collect_mask"].tolist() == [False, False, True, False]
        assert (
            scored_batch.non_tensor_batch[TRACE_KEY][2]["kind"] == "draft_branch_replay"
        )
        collected.append(scored_batch)
        return namespace["_speco_collect_oldlogprob_features"](
            trainer, scored_batch, capture_plan, output
        )

    def execute_collection(collection_plan, payload):
        dispatched.extend(payload.samples)
        return SimpleNamespace(collected_samples=len(payload.samples))

    trainer = SimpleNamespace(
        global_steps=2,
        tokenizer=SimpleNamespace(pad_token_id=0),
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(strategy="fsdp2"))
        ),
        actor_rollout_wg=SimpleNamespace(compute_log_prob=actor_forward),
        _speco_oldlogprob_collection_enabled=lambda: True,
        _speco_rollout_idle_worker_enabled=lambda: False,
        _speco_plan_drafter_collection=lambda source: SimpleNamespace(
            collect=True, collect_interval_matched=True
        ),
        _speco_log_drafter_collection_plan=lambda plan: None,
        _speco_build_oldlogprob_collect_plan=lambda batch: plan,
        _speco_drafter_training_config=lambda: {
            "drafter_opd": {"enable": True, "replay_full_suffix": True}
        },
        _speco_oldlogprob_calculate_entropy=lambda: False,
        _speco_oldlogprob_aux_layer_ids=lambda: [0],
        _speco_oldlogprob_aux_layer_id_space=lambda: "decoder",
        _speco_oldlogprob_hidden_capture_impl=lambda: "forward_hook",
        _speco_oldlogprob_hidden_layout=lambda: "dflash_aux_plus_last",
        _speco_oldlogprob_target_logz_temperature=lambda: None,
        _speco_collect_oldlogprob_features=collect,
        _speco_flatten_non_tensor_rows=lambda value: value,
        _speco_tensor_rows=lambda value: list(value) if value is not None else [],
        _speco_sequence_item=lambda value, index: None
        if value is None
        else value[index],
        _speco_sum_timing_rows=lambda value: None,
        _speco_dispatch_bucket_count=lambda: 1,
        _speco_get_drafter_scheduler=lambda: SimpleNamespace(
            prepare_collection_payload=lambda **kwargs: SimpleNamespace(**kwargs)
        ),
        _speco_execute_collection=execute_collection,
    )
    path = (
        Path(__file__).resolve().parents[2] / "verl_speco/trainer/speco_ray_trainer.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hook = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "compute_old_log_prob_with_speco"
    )
    namespace = dict(vars(oldlogprob_runtime))
    namespace.update(
        self=trainer,
        time=time,
        torch=torch,
        DataProto=DataProto,
        tu=tu,
        DrafterCollectionSource=DrafterCollectionSource,
        _select_policy_model_batch=lambda batch: batch,
        _get_nested=lambda config, path, default=None: default,
        left_right_2_no_padding=left_right_2_no_padding,
        no_padding_2_padding=no_padding_2_padding,
        uuid=uuid,
        Any=Any,
        cast=cast,
        detach_student_feature=lambda value: value.detach(),
    )
    collector = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_speco_collect_oldlogprob_features"
    )
    exec(
        compile(ast.Module(body=[collector], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    exec(
        compile(ast.Module(body=[hook], type_ignores=[]), str(path), "exec"), namespace
    )
    result, mfu = namespace[hook.name](trainer, original)
    assert len(actor_calls) == len(collected) == 1
    assert len(result) == len(original) == 2
    width = original.batch["responses"].shape[1]
    assert result.batch["old_log_probs"].shape == (2, width)
    torch.testing.assert_close(
        result.batch["old_log_probs"],
        original.batch["input_ids"][:, 2 : 2 + width].float(),
    )
    assert mfu == 0.5
    assert len(dispatched) == 1
    assert dispatched[0]["responses"].tolist() == [[4, 5, 8]]
    assert dispatched[0]["hidden_positions"].tolist() == [list(range(6))]
    assert dispatched[0]["hidden_states"].shape == (1, 6, 12)
