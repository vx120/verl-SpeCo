import functools
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
