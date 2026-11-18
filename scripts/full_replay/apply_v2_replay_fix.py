"""Apply the full-replay V2 transport hotfix without changing Git's index."""
from pathlib import Path
import ast
import sys
import time

root = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/model/ljh/draaft-opd_VAT/verl-SpeCo/verl_speco")
runtime = root / "integration/vllm_runtime.py"
original = runtime.read_text(encoding="utf-8")
text = original
hooks = [
    ("    async def run_server(self, args):\n", "enforce_replay_engine_args", "args", 8),
    ("    async def run_server_with_speco_stats(self, args):\n", "enforce_replay_engine_args", "args", 8),
    ('def install_vllm_runtime_observability() -> bool:\n    """Install lightweight vLLM logging hooks for all rollout modes."""\n',
     "patch_v2_draft_token_copy", "", 4),
    ("def _speco_vllm_worker_main_with_runtime_observability(*args, **kwargs):\n",
     "patch_v2_draft_token_copy", "", 4),
]
for marker, name, argument, indent in hooks:
    if text.count(marker) != 1:
        raise SystemExit("Unexpected runtime version; no files changed: " + marker)
    pad = " " * indent
    body = f"{pad}from verl_speco.integration.opd_vllm_tokens import {name}\n\n{pad}{name}({argument})\n"
    rest = text.split(marker, 1)[1]
    if rest.startswith(body) or rest.startswith("\n" + body):
        continue
    text = text.replace(marker, marker + body, 1)
MODULE = '''import functools
import json
import logging
from verl_speco.integration.drafter_config_env import get_drafter_config_env

logger = logging.getLogger(__name__)

def full_replay_enabled():
    config = json.loads(get_drafter_config_env() or "{}")
    opd = config.get("training", {}).get("drafter_opd", {})
    return bool(config.get("enable") and opd.get("enable") and opd.get("replay_full_suffix"))

def enforce_replay_engine_args(args):
    if full_replay_enabled():
        if getattr(args, "pipeline_parallel_size", 1) != 1:
            raise ValueError("Draft-OPD full replay requires pipeline_parallel_size=1")
        args.async_scheduling = False
        logger.warning("[speco opd replay] forcing engine async_scheduling=False after CLI parsing")

class _CopyRequiredBatch:
    has_structured_output_reqs = True
    def __init__(self, batch):
        self._batch = batch
    def __getattr__(self, name):
        return getattr(self._batch, name)

def patch_v2_draft_token_copy(handler_class=None):
    if not full_replay_enabled():
        return False
    if handler_class is None:
        try:
            from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
        except ImportError:
            return False
        handler_class = DraftTokensHandler
    original = handler_class.set_draft_tokens
    if getattr(original, "_speco_full_replay_copy", False):
        return True
    @functools.wraps(original)
    def set_draft_tokens(self, input_batch, draft_tokens):
        if full_replay_enabled() and not input_batch.has_structured_output_reqs:
            input_batch = _CopyRequiredBatch(input_batch)
        return original(self, input_batch, draft_tokens)
    set_draft_tokens._speco_full_replay_copy = True
    handler_class.set_draft_tokens = set_draft_tokens
    logger.warning("[speco opd replay] V2 real proposal CPU copy installed")
    return True
'''
ast.parse(text)
ast.parse(MODULE)
module = root / "integration/opd_vllm_tokens.py"
for target, content in [(module, MODULE), (runtime, text)]:
    if target.exists():
        old = target.read_text(encoding="utf-8")
        if old == content:
            continue
        backup = target.with_name(target.name + f".before_opd_v2_fix.{time.time_ns()}")
        backup.write_bytes(target.read_bytes())
    target.write_text(content, encoding="utf-8", newline="\n")
    print("Updated:", target)
import runpy
second = Path(__file__).with_name("apply_v2_replay_cuda_fix.py")
if second.exists():
    runpy.run_path(str(second), run_name="__main__")
else:
    raise SystemExit("Also copy apply_v2_replay_cuda_fix.py and rerun this installer.")
