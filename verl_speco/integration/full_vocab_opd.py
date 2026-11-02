"""Small-scale full-vocabulary OPD with unquantized, bias-free output heads."""

from typing import Any

import torch
import torch.nn.functional as F


async def score_teacher_hidden(
    engine: Any, prompt_ids: list[int], response_ids: list[int], request_id: str
):
    """Return final hidden rows predicting the supplied response token IDs."""
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    start = len(prompt_ids) - 1
    params = SamplingParams(
        temperature=0,
        max_tokens=1,
        output_kind=RequestOutputKind.FINAL_ONLY,
        extra_args={
            "hidden_state_capture": {
                "start": start,
                "end": start + len(response_ids),
                "coordinate": "absolute",
                "min_rows": len(response_ids),
                "collection_id": request_id,
                "hidden_layout": "final",
            }
        },
    )
    async for output in engine.generate(
        {"prompt_token_ids": prompt_ids + response_ids}, params, request_id
    ):
        if output.finished:
            aux = output.aux_outputs
            if aux is None or aux.hidden_state_capture is None:
                raise RuntimeError(f"Teacher hidden capture failed: {aux}")
            return aux.hidden_state_capture
    raise RuntimeError("Teacher stream ended without a final output")


async def score_teacher_payload(
    engine: Any,
    prompt_ids: list[int],
    response_ids: list[int],
    request_id: str,
    *,
    teacher_model: str,
    transport_config: dict | None = None,
    global_step: int = 0,
    replica_rank: int = 0,
    chunk_rows: int = 256,
) -> dict:
    """Score supplied tokens and deliver aligned hidden rows plus a logits contract.

    The caller owns the TransferQueue lifetime and releases the returned chunk
    references only after all readers finish. No optimizer or training is run.
    """
    from dataclasses import asdict

    from verl_speco.integration.live_hidden_transport import put_tensor_chunks

    if not response_ids:
        raise ValueError("response_ids must not be empty")
    result = await score_teacher_hidden(engine, prompt_ids, response_ids, request_id)
    positions = torch.from_numpy(result.hidden_positions.copy())
    start = len(prompt_ids) - 1
    if not torch.equal(positions, torch.arange(start, start + len(response_ids))):
        raise ValueError("Teacher hidden positions do not align with response tokens")
    if result.logits_contract is None:
        raise ValueError("Teacher did not provide a logits reconstruction contract")
    payload = {
        "schema_version": 1,
        "request_id": request_id,
        "collection_id": request_id,
        "teacher_model": teacher_model,
        "hidden_layout": result.hidden_layout,
        "includes_final_layer": result.includes_final_layer,
        "layer_ids": list(result.layer_ids),
        "prompt_token_ids": list(prompt_ids),
        "global_step": global_step,
        "replica_rank": replica_rank,
        "segments": [asdict(segment) for segment in result.segments],
        "logits_contract": dict(result.logits_contract),
        "tensors": {
            **result.tensors,
            "hidden_positions": positions,
            "response_token_ids": torch.tensor(response_ids, dtype=torch.long),
        },
    }
    return put_tensor_chunks(payload, transport_config or {}, chunk_rows=chunk_rows)


def reverse_kl_from_hidden(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    teacher_contract: dict,
    *,
    chunk_rows: int = 32,
) -> torch.Tensor:
    """Mean KL(student || teacher) over valid rows selected by the caller.

    Weights must use the same vocabulary order. This initial path targets
    standard bias-free heads; custom logits transforms require their own path.
    The caller controls student temperature before calling a custom loss path.
    Teacher tensors receive no gradient. Chunking bounds projection sizes;
    autograd still retains student intermediates until backward.
    """
    total = student_hidden.new_zeros((), dtype=torch.float32)
    vocab = teacher_contract["vocab_size"]
    for start in range(0, student_hidden.shape[0], chunk_rows):
        end = start + chunk_rows
        with torch.no_grad():
            logits = F.linear(teacher_hidden[start:end], teacher_weight)
            cap = teacher_contract["soft_cap"]
            if cap is not None:
                logits = cap * torch.tanh(logits / cap)
            logits = logits[..., :vocab] * teacher_contract["logit_scale"]
            logq = logits.float().log_softmax(-1)
        logp = (
            F.linear(student_hidden[start:end], student_weight)[..., :vocab]
            .float()
            .log_softmax(-1)
        )
        total = total + (logp.exp() * (logp - logq)).sum()
    return total / student_hidden.shape[0]
