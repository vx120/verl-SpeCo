# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import random

import pytest

from verl_speco.trainer.sample_cycle import sample_cyclic_without_replacement


@pytest.mark.parametrize("hard_ratio", [0.0, 0.2])
def test_cycle_sampler_covers_pool_before_reuse(hard_ratio: float) -> None:
    samples = [
        {"sample_index": index, "accept_len": float(index % 7)}
        for index in range(128)
    ]
    seen_ids: set[int] = set()

    draws = []
    for optimizer_step in range(20):
        rng = random.Random(optimizer_step)

        def sample_once(remaining, take):
            hard_count = round(take * hard_ratio)
            hard = sorted(remaining, key=lambda item: item["accept_len"])[
                :hard_count
            ]
            hard_ids = {id(item) for item in hard}
            random_pool = [item for item in remaining if id(item) not in hard_ids]
            return hard + rng.sample(random_pool, take - len(hard))

        batch = sample_cyclic_without_replacement(
            samples,
            8,
            seen_ids,
            sample_once,
        )
        assert len({id(item) for item in batch}) == 8
        draws.extend(batch)

    assert len(draws) == 160
    assert len({id(item) for item in draws[:128]}) == 128
    assert len({id(item) for item in draws}) == 128


def test_cycle_sampler_does_not_duplicate_across_epoch_boundary_batch() -> None:
    samples = [{"sample_index": index} for index in range(10)]
    seen_ids = {id(item) for item in samples[:7]}
    rng = random.Random(0)

    batch = sample_cyclic_without_replacement(
        samples,
        6,
        seen_ids,
        lambda remaining, take: rng.sample(remaining, take),
    )

    assert len(batch) == 6
    assert len({id(item) for item in batch}) == 6
    assert {item["sample_index"] for item in samples[7:]}.issubset(
        {item["sample_index"] for item in batch}
    )
