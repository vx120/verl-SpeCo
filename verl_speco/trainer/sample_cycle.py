# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Dependency-free sampling helpers for online drafter training."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


SampleT = TypeVar("SampleT")


def sample_cyclic_without_replacement(
    available_data: list[SampleT],
    batch_size: int,
    seen_ids: set[int],
    sample_once: Callable[[list[SampleT], int], list[SampleT]],
) -> list[SampleT]:
    """Build a batch while exhausting the current sample epoch before reuse."""
    if batch_size <= 0 or not available_data:
        return []
    if len(available_data) <= batch_size:
        return list(available_data)

    available_ids = {id(item) for item in available_data}
    seen_ids.intersection_update(available_ids)

    selected: list[SampleT] = []
    selected_ids: set[int] = set()
    while len(selected) < batch_size:
        remaining = [
            item
            for item in available_data
            if id(item) not in seen_ids and id(item) not in selected_ids
        ]
        if not remaining:
            seen_ids.clear()
            continue

        take = min(batch_size - len(selected), len(remaining))
        chosen = sample_once(remaining, take)
        if len(chosen) != take:
            raise ValueError(
                f"sample_once returned {len(chosen)} samples; expected {take}"
            )
        selected.extend(chosen)
        chosen_ids = {id(item) for item in chosen}
        selected_ids.update(chosen_ids)
        seen_ids.update(chosen_ids)

    return selected
