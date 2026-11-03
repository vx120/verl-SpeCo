import pytest
import torch
import torch.nn.functional as F

from verl_speco.integration.full_vocab_opd import reverse_kl_from_hidden


@pytest.mark.parametrize("cap", [None, 2.0])
def test_chunked_reverse_kl_matches_dense_loss_and_gradients(cap):
    torch.manual_seed(42)
    hidden = torch.randn(5, 4, requires_grad=True)
    teacher = torch.randn(5, 4, requires_grad=True)
    weight = torch.randn(8, 4, requires_grad=True)
    teacher_weight = torch.randn(8, 4, requires_grad=True)
    contract = {"vocab_size": 7, "soft_cap": cap, "logit_scale": 0.7}
    loss = reverse_kl_from_hidden(
        hidden, teacher, weight, teacher_weight, contract, chunk_rows=2
    )
    actual = torch.autograd.grad(loss, (hidden, weight), retain_graph=True)
    q = F.linear(teacher.detach(), teacher_weight.detach())
    if cap is not None:
        q = cap * torch.tanh(q / cap)
    logq = (q[:, :7] * 0.7).log_softmax(-1)
    logp = F.linear(hidden, weight)[:, :7].log_softmax(-1)
    reference = (logp.exp() * (logp - logq)).sum(-1).mean()
    expected = torch.autograd.grad(reference, (hidden, weight))
    torch.testing.assert_close(loss, reference)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    loss.backward()
    assert teacher.grad is None and teacher_weight.grad is None


@pytest.mark.asyncio
async def test_teacher_payload_aligns_response_tokens_and_contract(monkeypatch):
    from dataclasses import dataclass
    from types import SimpleNamespace

    import numpy as np

    from verl_speco.integration import full_vocab_opd as opd

    @dataclass
    class Segment:
        kind: str = "prompt"
        start_abs: int = 2
        end_abs: int = 4

    result = SimpleNamespace(
        hidden_positions=np.array([2, 3]),
        tensors={"hidden_states": torch.ones(2, 4, dtype=torch.bfloat16)},
        logits_contract={"vocab_size": 100, "logit_scale": 1.0, "soft_cap": None},
        segments=(Segment(),),
        hidden_layout="final",
        includes_final_layer=True,
        layer_ids=(),
    )

    async def score(engine, prompt, response, request_id):
        assert prompt == [10, 11, 12] and response == [13, 14]
        return result

    monkeypatch.setattr(opd, "score_teacher_hidden", score)
    payload = await opd.score_teacher_payload(
        None,
        [10, 11, 12],
        [13, 14],
        "r",
        teacher_model="teacher-v1",
        response_loss_mask=[True, False],
    )
    assert payload["tensors"]["response_loss_mask"].tolist() == [True, False]
    assert payload["teacher_model"] == "teacher-v1"
    assert payload["prompt_token_ids"] == [10, 11, 12]
    assert payload["tensors"]["response_token_ids"].tolist() == [13, 14]
    assert payload["tensors"]["hidden_positions"].tolist() == [2, 3]
    assert payload["logits_contract"] == result.logits_contract
    assert payload["segments"] == [{"kind": "prompt", "start_abs": 2, "end_abs": 4}]
    result.hidden_positions[1] = 8
    with pytest.raises(ValueError, match="align"):
        await opd.score_teacher_payload(
            None, [10, 11, 12], [13, 14], "r", teacher_model="teacher-v1"
        )


@pytest.mark.asyncio
async def test_rollout_masks_preserve_unsupervised_context_and_eos(monkeypatch):
    from types import SimpleNamespace

    from verl_speco.integration import full_vocab_opd as opd

    batch = {
        "prompts": torch.tensor([[0, 0, 10, 11], [0, 20, 21, 22]]),
        "responses": torch.tensor([[12, 13, 0], [23, 0, 0]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 0]]),
        "response_mask": torch.tensor([[1, 0, 1], [1, 1, 0]]),
    }
    original = {k: v.clone() for k, v in batch.items()}
    calls = []

    async def score(engine, prompt, response, request_id, **kwargs):
        calls.append((prompt, response, kwargs["response_loss_mask"]))
        return {"request_id": request_id}

    monkeypatch.setattr(opd, "score_teacher_payload", score)
    payloads = await opd.score_rollout_teacher_payloads(
        None, SimpleNamespace(batch=batch), "batch", teacher_model="teacher"
    )
    assert calls == [
        ([10, 11], [12, 13, 0], [True, False, True]),
        ([20, 21, 22], [23, 0], [True, True]),
    ]
    assert [p["sample_index"] for p in payloads] == [0, 1]
    for key, value in batch.items():
        assert torch.equal(value, original[key])


@pytest.mark.asyncio
async def test_rollout_scoring_releases_earlier_samples_on_failure(monkeypatch):
    from types import SimpleNamespace

    from verl_speco.integration import full_vocab_opd as opd
    from verl_speco.integration import live_hidden_transport as transport

    batch = {
        "prompts": torch.tensor([[10], [20]]),
        "responses": torch.tensor([[11], [21]]),
        "attention_mask": torch.ones(2, 2),
    }
    cleared = []

    async def score(engine, prompt, response, request_id, **kwargs):
        if request_id.endswith("-1"):
            raise RuntimeError("teacher failed")
        return {"request_id": request_id}

    monkeypatch.setattr(opd, "score_teacher_payload", score)
    monkeypatch.setattr(transport, "release_tensor_chunks", cleared.append)
    with pytest.raises(RuntimeError, match="teacher failed"):
        await opd.score_rollout_teacher_payloads(
            None, SimpleNamespace(batch=batch), "batch", teacher_model="teacher"
        )
    assert cleared == [{"request_id": "batch-0", "sample_index": 0}]
