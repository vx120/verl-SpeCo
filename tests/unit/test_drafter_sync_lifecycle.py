# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
from __future__ import annotations

import pytest

from verl_speco.trainer.drafter_lifecycle import SyncDrafterLifecycle


def test_drafter_publish_waits_for_student_rollout_weight_sync() -> None:
    events: list[tuple[str, object]] = []
    actor_output = {"student": "updated"}

    lifecycle = SyncDrafterLifecycle(
        defer_publish_until_rollout_weight_sync=True,
        publish_drafter=lambda trained: events.append(("publish_drafter", trained))
        or {"drafter/published": 1},
        update_output_metrics=lambda output, metrics: events.append(
            ("attach_metrics", (output, metrics))
        ),
    )

    events.append(("student_update", actor_output))
    assert (
        lifecycle.after_student_update(
            actor_output=actor_output,
            drafter_trained=True,
        )
        == {}
    )
    assert lifecycle.has_pending_publish
    assert events == [("student_update", actor_output)]

    events.append(("student_rollout_weight_sync", True))
    assert lifecycle.after_student_rollout_weight_sync() == {"drafter/published": 1}

    assert [name for name, _ in events] == [
        "student_update",
        "student_rollout_weight_sync",
        "publish_drafter",
        "attach_metrics",
    ]
    assert not lifecycle.has_pending_publish


def test_drafter_publish_is_immediate_without_weight_sync_boundary() -> None:
    published: list[bool] = []
    lifecycle = SyncDrafterLifecycle(
        defer_publish_until_rollout_weight_sync=False,
        publish_drafter=lambda trained: published.append(trained)
        or {"drafter/published": int(trained)},
        update_output_metrics=lambda output, metrics: None,
    )

    assert lifecycle.after_student_update(
        actor_output=object(), drafter_trained=True
    ) == {"drafter/published": 1}
    assert published == [True]
    assert not lifecycle.has_pending_publish


def test_second_student_update_before_weight_sync_fails_closed() -> None:
    lifecycle = SyncDrafterLifecycle(
        defer_publish_until_rollout_weight_sync=True,
        publish_drafter=lambda trained: {},
        update_output_metrics=lambda output, metrics: None,
    )
    lifecycle.after_student_update(actor_output=object(), drafter_trained=True)

    with pytest.raises(RuntimeError, match="previous Student rollout weight sync"):
        lifecycle.after_student_update(actor_output=object(), drafter_trained=True)
