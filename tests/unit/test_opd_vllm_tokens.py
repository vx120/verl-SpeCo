"""Exercise the installed V2 handler source without a GPU or vLLM imports."""

import ast
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verl_speco.integration import opd_vllm_tokens as replay


def configure(monkeypatch, enabled):
    monkeypatch.setenv("VERL_SPECO_DRAFTER_CONFIG", json.dumps({
        "enable": True, "training": {"drafter_opd": {
            "enable": True, "replay_full_suffix": enabled,
        }},
    }))


@pytest.fixture
def handler_type():
    source = Path(__file__).resolve().parents[3] / "vllm-main-20260702-dspark/vllm/v1/worker/gpu/spec_decode/utils.py"
    if not source.exists():
        pytest.skip("Local V2 source is unavailable")
    cls = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
               if isinstance(n, ast.ClassDef) and n.name == "DraftTokensHandler")
    stream = SimpleNamespace(wait_stream=lambda other: None)
    event = SimpleNamespace(record=lambda: None, synchronize=lambda: None)
    torch = SimpleNamespace(cuda=SimpleNamespace(
        Stream=lambda device: stream, current_stream=lambda device: stream,
        stream=lambda stream: nullcontext()), Event=lambda: event)
    ns = {"torch": torch, "async_copy_to_np": lambda tensor: tensor.values.copy(),
          "DraftTokenIds": lambda req_ids, ids: SimpleNamespace(req_ids=req_ids, draft_token_ids=ids)}
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
    return ns["DraftTokensHandler"]


@pytest.mark.parametrize("capacity", [7, 15])
@pytest.mark.parametrize("structured", [False, True])
def test_real_v2_handler_copies_complete_proposals(monkeypatch, handler_type, capacity, structured):
    configure(monkeypatch, True)
    values = np.arange(2 * capacity).reshape(2, capacity) + 100
    tensor = SimpleNamespace(values=values, shape=values.shape, record_stream=lambda stream: None)
    batch = SimpleNamespace(req_ids=["a", "b"], has_structured_output_reqs=structured)
    before = handler_type()
    before.set_draft_tokens(batch, tensor)
    # Reproduce upstream's non-structured placeholder path before the fix.
    assert before.get_draft_tokens().draft_token_ids == (values.tolist() if structured else [[-1] * capacity] * 2)
    assert replay.patch_v2_draft_token_copy(handler_type)
    installed = handler_type.set_draft_tokens
    assert replay.patch_v2_draft_token_copy(handler_type)
    assert handler_type.set_draft_tokens is installed
    after = handler_type()
    after.set_draft_tokens(batch, tensor)
    values[:] = -1  # CPU copy must own its data after the GPU proposal changes.
    output = after.get_draft_tokens()
    assert output.req_ids == ["a", "b"]
    assert output.draft_token_ids == np.arange(2 * capacity).reshape(2, capacity).__add__(100).tolist()
    assert batch.has_structured_output_reqs is structured


def test_disabled_mode_keeps_original_handler(monkeypatch, handler_type):
    configure(monkeypatch, False)
    original = handler_type.set_draft_tokens
    assert not replay.patch_v2_draft_token_copy(handler_type)
    assert handler_type.set_draft_tokens is original


def test_cli_drops_false_but_replay_restores_it(monkeypatch):
    configure(monkeypatch, True)
    source = Path(__file__).resolve().parents[3] / "Draft-OPD/verl/verl/workers/rollout/vllm_rollout/utils.py"
    if not source.exists():
        pytest.skip("Local upstream source is unavailable")
    fn = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
              if isinstance(n, ast.FunctionDef) and n.name == "build_cli_args_from_config")
    ns = {"Any": object, "json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    assert ns[fn.name]({"async_scheduling": False}) == []
    args = SimpleNamespace(async_scheduling=None, pipeline_parallel_size=1)
    replay.enforce_replay_engine_args(args)
    assert args.async_scheduling is False


def test_engine_args_disabled_and_unsupported_pp(monkeypatch):
    args = SimpleNamespace(async_scheduling=True, pipeline_parallel_size=2)
    configure(monkeypatch, False)
    replay.enforce_replay_engine_args(args)
    assert args.async_scheduling is True
    configure(monkeypatch, True)
    with pytest.raises(ValueError, match="pipeline_parallel_size=1"):
        replay.enforce_replay_engine_args(args)


def test_runtime_installs_worker_and_both_server_paths():
    source = Path(__file__).resolve().parents[2] / "verl_speco/integration/vllm_runtime.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    targets = {
        "_speco_vllm_worker_main_with_runtime_observability": "patch_v2_draft_token_copy",
        "install_vllm_runtime_observability": "patch_v2_draft_token_copy",
        "run_server": "enforce_replay_engine_args",
        "run_server_with_speco_stats": "enforce_replay_engine_args",
    }
    for name, call in targets.items():
        fn = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
        assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == call for n in ast.walk(fn))


def test_cuda_worker_entrypoint_installs_real_copy(monkeypatch, handler_type):
    import sys
    import types

    configure(monkeypatch, True)
    original = handler_type.set_draft_tokens
    token_module = types.ModuleType("vllm.v1.worker.gpu.spec_decode.utils")
    token_module.DraftTokensHandler = handler_type
    monkeypatch.setitem(sys.modules, token_module.__name__, token_module)
    calls = []

    class WorkerProc:
        @staticmethod
        def worker_main():
            calls.append(handler_type.set_draft_tokens is not original)
            return "worker-started"

    executor = types.ModuleType("vllm.v1.executor")
    executor.multiproc_executor = SimpleNamespace(WorkerProc=WorkerProc)
    monkeypatch.setitem(sys.modules, executor.__name__, executor)
    source = Path(__file__).resolve().parents[2] / "verl_speco/integration/vllm_runtime.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"patch_vllm_worker_proc_entrypoint", "_speco_vllm_worker_main_with_runtime_observability"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"_maybe_apply_vllm_ascend_global_patch": lambda: False,
          "patch_vllm_dflash_config_aliases": lambda: None,
          "patch_vllm_dspark_registry_aliases": lambda: None,
          "patch_vllm_dspark_runtime": lambda: None,
          "install_vllm_spec_decode_acceptance_logging": lambda: None}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), ns)
    # False means CUDA, so it must not disable the generic worker hook.
    assert ns["patch_vllm_worker_proc_entrypoint"]()
    assert WorkerProc.worker_main() == "worker-started"
    assert calls == [True]


@pytest.mark.parametrize("capacity", [7, 15])
@pytest.mark.parametrize("prior_outputs", [[], [8, 9]])
def test_new_and_resumed_cuda_padding_has_no_replay_tokens(capacity, prior_outputs):
    from verl_speco.integration.verify_trace import snapshot_verify_step, finish_verify_step, validate_verify_trace

    request = SimpleNamespace(prompt_token_ids=[1, 2], _output_token_ids=prior_outputs.copy(), is_finished=lambda: False)
    scheduler = SimpleNamespace(requests={"r": request})
    scheduled = SimpleNamespace(num_scheduled_tokens={"r": capacity + 1},
        scheduled_spec_decode_tokens={"r": [-1] * capacity}, num_invalid_spec_tokens=None,
        scheduled_new_reqs=[SimpleNamespace(req_id="r")])
    output = SimpleNamespace(req_id_to_index={"r": 0}, sampled_token_ids=[[40]])
    snapshot = snapshot_verify_step(scheduler, scheduled, output, capture_draft_tokens=True)
    assert snapshot[0][3:] == (0, 1, [])
    request._output_token_ids.append(40)
    trace = finish_verify_step(snapshot, layout="dspark", capacity=capacity)["r"]
    # Seed earlier commits to validate the resumed branch's full coordinates.
    if prior_outputs:
        trace["prompt_length"] += len(prior_outputs)
        trace["response_length"] -= len(prior_outputs)
    validate_verify_trace(trace, layout="dspark", capacity=capacity)
    assert trace["steps"][0]["draft_token_ids"] == []
    assert trace["steps"][0]["drafts"] == 0


@pytest.mark.parametrize("tokens,generated,admitted", [
    ([-1, -1], 1, False), ([-1, -1], 2, True), ([4, -1], 1, True),
])
def test_bad_real_proposals_remain_fail_closed(tokens, generated, admitted):
    from verl_speco.integration.verify_trace import snapshot_verify_step

    request = SimpleNamespace(prompt_token_ids=[1], _output_token_ids=[], is_finished=lambda: False)
    scheduled = SimpleNamespace(num_scheduled_tokens={"r": 3},
        scheduled_spec_decode_tokens={"r": tokens}, num_invalid_spec_tokens=None,
        scheduled_new_reqs=[SimpleNamespace(req_id="r")] if admitted else [])
    with pytest.raises(ValueError, match="request='r'"):
        snapshot_verify_step(SimpleNamespace(requests={"r": request}), scheduled,
            SimpleNamespace(req_id_to_index={"r": 0}, sampled_token_ids=[[5] * generated]),
            capture_draft_tokens=True)


def test_worker_extension_installs_before_model_construction():
    source = Path(__file__).resolve().parents[2] / "verl_speco/integration/vllm_runtime.py"
    cls = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
               if isinstance(n, ast.ClassDef) and n.name == "SpecoVLLMColocateWorkerExtension")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__new__")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "patch_v2_draft_token_copy" for n in ast.walk(fn))
