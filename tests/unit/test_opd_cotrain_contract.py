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

from verl_speco.integration.opd_cotrain import (
    detach_student_feature,
    ensure_student_only_feature_payload,
    opd_cotrain_enabled,
    student_policy_batch_keys,
    validate_sync_opd_cotrain_config,
)


def _config(
    *,
    opd: bool = True,
    drafter: bool = True,
    train_drafter: bool = True,
    strategy: str = "fsdp2",
) -> dict:
    return {
        "actor_rollout_ref": {
            "actor": {"strategy": strategy},
            "rollout": {
                "drafter": {
                    "enable": drafter,
                    "enable_drafter_training": train_drafter,
                    "training": {
                        "mode": "online",
                        "collect_hidden_states_from_old_logprob": True,
                        "collect_hidden_states_from_sgl": False,
                        "use_logits": False,
                    },
                }
            },
        },
        "algorithm": {"rollout_correction": {"bypass_mode": False}},
        "distillation": {
            "enabled": opd,
            "teacher_models": {
                "teacher_model": {
                    "model_path": "/teacher",
                    "inference": {"name": "vllm"},
                }
            },
        },
    }


@pytest.mark.parametrize(
    ("opd", "drafter", "train_drafter", "expected_cotrain"),
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
def test_supported_opd_drafter_configurations(
    opd: bool, drafter: bool, train_drafter: bool, expected_cotrain: bool
) -> None:
    config = _config(
        opd=opd,
        drafter=drafter,
        train_drafter=train_drafter,
    )

    validate_sync_opd_cotrain_config(config)

    assert opd_cotrain_enabled(config) is expected_cotrain


def test_cotrain_requires_fsdp2() -> None:
    with pytest.raises(ValueError, match="strategy=fsdp2"):
        validate_sync_opd_cotrain_config(_config(strategy="fsdp"))


def test_cotrain_requires_oldlogprob_forward() -> None:
    config = _config()
    config["actor_rollout_ref"]["rollout"]["drafter"]["training"][  # type: ignore[index]
        "collect_hidden_states_from_old_logprob"
    ] = False

    with pytest.raises(ValueError, match="collect_hidden_states_from_old_logprob=true"):
        validate_sync_opd_cotrain_config(config)


def test_cotrain_rejects_oldlogprob_bypass() -> None:
    config = _config()
    config["algorithm"]["rollout_correction"]["bypass_mode"] = True  # type: ignore[index]

    with pytest.raises(ValueError, match="bypass_mode must be false"):
        validate_sync_opd_cotrain_config(config)


def test_teacher_inference_rejects_drafter_config() -> None:
    config = _config(drafter=False, train_drafter=False)
    teacher = config["distillation"]["teacher_models"]["teacher_model"]  # type: ignore[index]
    teacher["inference"]["drafter"] = {"enable": True}  # type: ignore[index]

    with pytest.raises(ValueError, match="Teacher speculative decoding"):
        validate_sync_opd_cotrain_config(config)


def test_student_feature_forward_excludes_teacher_tensors() -> None:
    keys = student_policy_batch_keys(
        [
            "input_ids",
            "attention_mask",
            "teacher_ids",
            "teacher_logprobs",
            "responses",
        ]
    )

    assert keys == ["input_ids", "attention_mask", "responses"]


def test_drafter_worker_boundary_rejects_teacher_tensors() -> None:
    ensure_student_only_feature_payload({"hidden_states": object()})

    with pytest.raises(ValueError, match="teacher_logprobs"):
        ensure_student_only_feature_payload(
            {"hidden_states": object(), "teacher_logprobs": object()}
        )


def test_student_and_drafter_gradients_are_isolated() -> None:
    torch = pytest.importorskip("torch")
    student = torch.nn.Linear(3, 4, bias=False)
    drafter = torch.nn.Linear(4, 2, bias=False)
    inputs = torch.randn(2, 3)
    teacher_target = torch.randn(2, 4)

    student_features = student(inputs)
    drafter_features = detach_student_feature(student_features)
    drafter_loss = drafter(drafter_features).square().mean()
    drafter_loss.backward()

    assert all(parameter.grad is None for parameter in student.parameters())
    assert all(parameter.grad is not None for parameter in drafter.parameters())

    student.zero_grad(set_to_none=True)
    drafter.zero_grad(set_to_none=True)
    student_opd_loss = (student_features - teacher_target).square().mean()
    student_opd_loss.backward()

    assert all(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in drafter.parameters())
