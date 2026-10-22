# Copyright 2026 Individual Contributor: vx120
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

import asyncio
from types import SimpleNamespace
from verl_speco.integration.vllm_runtime import (
    _canonicalize_tied_weight_bucket,
    _SpecoTiedTargetWeightCompat,
    _apply_vllm_boolean_launch_overrides,
)


def test_tied_weight_bucket_rewrites_standalone_lm_head_alias() -> None:
    weights = [("layers.0.weight", 1), ("lm_head.weight", 2)]

    assert _canonicalize_tied_weight_bucket(weights, tied=True) == [
        ("layers.0.weight", 1),
        ("model.embed_tokens.weight", 2),
    ]


def test_tied_weight_bucket_drops_alias_when_canonical_weight_is_present() -> None:
    weights = [
        ("lm_head.weight", 1),
        ("model.embed_tokens.weight", 2),
    ]

    assert _canonicalize_tied_weight_bucket(weights, tied=True) == [
        ("model.embed_tokens.weight", 2)
    ]


def test_untied_weight_bucket_is_unchanged() -> None:
    weights = [("lm_head.weight", 1)]

    assert _canonicalize_tied_weight_bucket(weights, tied=False) is weights


def test_target_weight_update_canonicalizes_tied_alias_before_upstream_loader() -> None:
    calls = []
    shared_weight = object()

    class BaseExtension:
        def _update_weights(self, weights, peft_config, base_sync_done):
            calls.append((weights, peft_config, base_sync_done))
            return "loaded"

    class Extension(_SpecoTiedTargetWeightCompat, BaseExtension):
        model_runner = SimpleNamespace(
            model=SimpleNamespace(
                lm_head=SimpleNamespace(weight=shared_weight),
                model=SimpleNamespace(
                    embed_tokens=SimpleNamespace(weight=shared_weight)
                ),
            )
        )

    assert (
        Extension()._update_weights(
            [("lm_head.weight", 7)], peft_config=None, base_sync_done=False
        )
        == "loaded"
    )
    assert calls == [
        ([("model.embed_tokens.weight", 7)], None, False),
    ]


def test_explicit_false_rollout_flags_survive_cli_conversion():
    args = SimpleNamespace(
        enable_prefix_caching=True, enable_chunked_prefill=True, async_scheduling=True
    )
    _apply_vllm_boolean_launch_overrides(
        args,
        {
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "engine_kwargs": {"vllm": {"async_scheduling": False}},
        },
    )
    assert vars(args) == {
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "async_scheduling": False,
    }


def test_server_receives_drafter_config_without_inherited_environment(monkeypatch):
    import json
    from verl_speco.integration import vllm_runtime as runtime

    class UpstreamServer:
        config = {}

        async def launch_server(self):
            return "launched"

    payload = json.dumps(
        {
            "enable": True,
            "speculative_algorithm": "DSPARK",
            "training": {"collect_hidden_states_from_vllm": True},
        }
    )
    monkeypatch.setenv(runtime.SPECO_DRAFTER_CONFIG_ENV, payload)
    server_class = runtime._build_speco_vllm_http_server_class(
        SimpleNamespace(vLLMHttpServer=UpstreamServer)
    )
    monkeypatch.delenv(runtime.SPECO_DRAFTER_CONFIG_ENV)
    monkeypatch.setattr(runtime, "install_vllm_runtime_observability", lambda: None)
    monkeypatch.setattr(
        runtime, "_ensure_vllm_drafter_speculative_config_from_env", lambda config: None
    )
    server = server_class()
    assert asyncio.run(server.launch_server()) == "launched"
    assert server._speco_vllm_hidden_capture_enabled is True
    assert server._speco_initial_draft_weights_required is True


def test_cpu_weight_handle_is_not_rewritten_as_cuda_handle():
    import torch
    from torch.multiprocessing.reductions import reduce_tensor
    from verl_speco.integration.vllm_runtime import _speco_rebuild_ipc_compat

    source = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    restored = _speco_rebuild_ipc_compat(reduce_tensor(source), device_id=0)
    assert restored.device.type == "cpu"
    assert torch.equal(restored, source)


def test_pause_compat_preserves_abort_wait_and_keep():
    from verl_speco.integration.vllm_runtime import (
        _install_vllm_pause_generation_compat,
    )

    calls = []

    async def pause(*, mode="abort", clear_cache=True):
        calls.append((mode, clear_cache))

    engine = SimpleNamespace(pause_generation=pause)
    _install_vllm_pause_generation_compat(engine)
    wrapped = engine.pause_generation
    _install_vllm_pause_generation_compat(engine)
    assert engine.pause_generation is wrapped
    asyncio.run(engine.pause_generation(wait_for_inflight_requests=False))
    asyncio.run(
        engine.pause_generation(wait_for_inflight_requests=True, clear_cache=False)
    )
    asyncio.run(engine.pause_generation(mode="keep", clear_cache=False))
    assert calls == [("abort", True), ("wait", False), ("keep", False)]
