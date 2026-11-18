# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Teacher-force recorded proposal branches through the actor collection RPC.

Original policy rows stay first. Added rows are exclusively drafter supervision;
the caller must strip them from the old-logprob result returned to PPO.
"""

import copy

import numpy as np
import torch

from verl_speco.integration.verify_trace import (
    TRACE_KEY,
    token_digest,
    validate_verify_trace,
)


def expand_replay_batch(batch, plan, *, max_blocks, pad_token_id=0, seed=0):
    """Return tensor/metadata payloads and a branch-only capture plan.

    Each branch is a separate causal sequence with its complete original prefix.
    No KV cache from the corrected continuation is reused. Sampling limits anchor
    count only; every selected proposal is scored through its last token.
    """
    if max_blocks < 0:
        raise ValueError("replay_max_blocks_per_sample must be non-negative (0 = all)")
    tensors = batch.batch
    # verl's text-only agent loop also carries a per-row multimodal column
    # containing {} or None. Only actual nonempty payloads are unsupported.
    multimodal = batch.non_tensor_batch.get("multi_modal_inputs")
    items = [multimodal] if isinstance(multimodal, dict) else multimodal
    if items is not None and any(
        item is not None and (not isinstance(item, dict) or len(item) > 0)
        for item in items
    ):
        raise ValueError("Draft-OPD branch replay does not support nonempty multimodal inputs")
    if tensors["position_ids"].ndim != 2:
        raise ValueError(
            "Draft-OPD branch replay requires text position_ids shaped [batch, sequence]; "
            f"got shape={tuple(tensors['position_ids'].shape)}"
        )
    prompts, responses = tensors["prompts"], tensors["responses"]
    count, prompt_width = prompts.shape
    response_width = responses.shape[1]
    traces = batch.non_tensor_batch.get(TRACE_KEY)
    if traces is None:
        raise ValueError("Draft-OPD full replay requires proposal traces")
    rows = []
    metadata = []
    owners = []
    for row in range(count):
        prompt_mask = tensors["attention_mask"][row, :prompt_width].bool()
        response_mask = tensors["attention_mask"][row, prompt_width:].bool()
        prefix = prompts[row, prompt_mask]
        answer = responses[row, response_mask]
        ids = torch.cat((prefix, answer))
        positions = tensors["position_ids"][row, tensors["attention_mask"][row].bool()]
        rows.append((prefix, answer, positions))
        metadata.append(traces[row])
        owners.append(-1)
        if not bool(plan["collect_mask"][row]):
            continue
        trace = traces[row]
        if trace is None or trace.get("kind") == "draft_branch_replay":
            raise ValueError("Draft-OPD replay requires original scheduler evidence")
        candidates = []
        for turn in trace.get("turns", [trace]):
            validate_verify_trace(
                turn, layout=turn["layout"], capacity=turn["num_spec_tokens"]
            )
            end = turn["prompt_length"] + turn["response_length"]
            if end > ids.numel() or (
                turn.get("prefix_digest") is not None
                and token_digest(ids[:end].tolist()) != turn["prefix_digest"]
            ):
                raise ValueError(
                    "Draft-OPD replay trace does not match the policy trajectory"
                )
            for step in turn["steps"]:
                if not step["drafts"] or not step["commit"] or step["truncated"]:
                    continue  # Stop/length-truncated outcomes are not complete blocks.
                proposals = step.get("draft_token_ids")
                if proposals is None:
                    raise ValueError(
                        "Draft-OPD full replay trace is missing proposal token IDs; restart rollout workers"
                    )
                anchor, accepted = step["anchor"], step["accepted"]
                if anchor < 0 or anchor >= ids.numel():
                    raise ValueError(
                        "Draft-OPD replay anchor is outside the policy trajectory"
                    )
                if (
                    ids[anchor + 1 : anchor + 1 + accepted].tolist()
                    != proposals[:accepted]
                ):
                    raise ValueError(
                        "Draft-OPD accepted proposal prefix differs from committed tokens"
                    )
                candidates.append((turn, step))
        if max_blocks and len(candidates) > max_blocks:
            generator = torch.Generator().manual_seed(seed + row)
            indices = torch.randperm(len(candidates), generator=generator)[
                :max_blocks
            ].tolist()
            candidates = [candidates[index] for index in sorted(indices)]
        for turn, step in candidates:
            anchor = step["anchor"]
            proposed = torch.tensor(
                step["draft_token_ids"], dtype=ids.dtype, device=ids.device
            )
            branch_prefix = ids[: anchor + 1]
            branch_positions = torch.cat(
                (
                    positions[: anchor + 1],
                    positions[anchor]
                    + torch.arange(1, proposed.numel() + 1, device=positions.device),
                )
            )
            rows.append((branch_prefix, proposed, branch_positions))
            metadata.append(
                {
                    "kind": "draft_branch_replay",
                    "layout": turn["layout"],
                    "num_spec_tokens": turn["num_spec_tokens"],
                    "prompt_length": anchor + 1,
                    "response_length": proposed.numel(),
                    "anchor": anchor,
                    "accepted": step["accepted"],
                    "draft_token_ids": step["draft_token_ids"],
                    "source_step": copy.deepcopy(step),
                }
            )
            owners.append(int(plan["owner_rank"][row]))
    # Branches were interleaved above; restore the policy rows as a leading slice.
    policy_indices = [index for index, owner in enumerate(owners) if owner < 0]
    branch_indices = [index for index, owner in enumerate(owners) if owner >= 0]
    order = policy_indices + branch_indices
    if not branch_indices:
        return None
    rows = [rows[index] for index in order]
    metadata = [metadata[index] for index in order]
    owners = [owners[index] for index in order]
    branch_count = len(branch_indices)
    # The original batch was dispatchable: a multiple of its size also is.
    while len(rows) % count:
        rows.append(rows[0])
        metadata.append(metadata[0])
        owners.append(-1)
    prompt_width = max(prompt_width, max(prefix.numel() for prefix, _, _ in rows))
    response_width = max(response_width, max(answer.numel() for _, answer, _ in rows))
    new_count = len(rows)
    out_prompts = prompts.new_full((new_count, prompt_width), pad_token_id)
    out_responses = responses.new_full((new_count, response_width), pad_token_id)
    attention = tensors["attention_mask"].new_zeros(
        (new_count, prompt_width + response_width)
    )
    position_ids = tensors["position_ids"].new_zeros(attention.shape)
    response_masks = attention.new_zeros(out_responses.shape)
    capture = torch.zeros(new_count, dtype=torch.bool)
    hidden_rows = plan["hidden_positions"].shape[1]
    hidden_positions = torch.zeros(new_count, hidden_rows, dtype=torch.long)
    hidden_mask = torch.zeros_like(hidden_positions, dtype=torch.bool)
    for row, (prefix, answer, positions) in enumerate(rows):
        p, r = prefix.numel(), answer.numel()
        out_prompts[row, prompt_width - p :] = prefix
        out_responses[row, :r] = answer
        attention[row, prompt_width - p : prompt_width + r] = 1
        position_ids[row, prompt_width - p : prompt_width] = positions[:p]
        position_ids[row, prompt_width : prompt_width + r] = positions[p:]
        if count <= row < count + branch_count:
            if hidden_rows <= r:
                raise ValueError(
                    "Draft-OPD replay capture window must include the anchor and all draft tokens"
                )
            response_masks[row, :r] = 1
            capture[row] = True
            start = max(0, p + r - hidden_rows)
            selected = torch.arange(start, p + r)
            hidden_positions[row, : selected.numel()] = selected
            hidden_mask[row, : selected.numel()] = True
        elif row < count and "response_mask" in tensors:
            response_masks[row, : tensors["response_mask"].shape[1]] = tensors[
                "response_mask"
            ][row]
    replay_plan = dict(plan)
    replay_plan.update(
        collect_mask=capture,
        hidden_positions=hidden_positions,
        hidden_position_mask=hidden_mask,
        owner_rank=torch.tensor(owners),
        prompt_lens=[prefix.numel() for prefix, _, _ in rows],
        response_lens=[answer.numel() for _, answer, _ in rows],
        selected_count=branch_count,
        candidate_count=branch_count,
        hard_enabled=False,
        request_ids=[None] * new_count,
        request_completion_indices=[None] * new_count,
        request_elapsed_secs=[None] * new_count,
    )
    return (
        {
            "prompts": out_prompts,
            "responses": out_responses,
            "input_ids": torch.cat((out_prompts, out_responses), dim=1),
            "attention_mask": attention,
            "position_ids": position_ids,
            "response_mask": response_masks,
        },
        {TRACE_KEY: np.asarray(metadata, dtype=object)},
        replay_plan,
    )
