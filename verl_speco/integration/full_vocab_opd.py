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
