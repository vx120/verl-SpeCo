#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
"""Second-stage full replay fix: CUDA worker hooks and synthetic padding."""
from pathlib import Path
import ast
import sys
import time

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/model/ljh/draaft-opd_VAT/verl-SpeCo/verl_speco")
if not (root / "integration/opd_vllm_tokens.py").exists():
    raise SystemExit("Apply the previous V2 token-copy hotfix first.")
runtime = root / "integration/vllm_runtime.py"
trace = root / "integration/verify_trace.py"
s = runtime.read_text(encoding="utf-8")
t = trace.read_text(encoding="utf-8")
a = "    if not _maybe_apply_vllm_ascend_global_patch():\n        return False\n    try:\n        from vllm.v1.executor import multiproc_executor"
b = "    _maybe_apply_vllm_ascend_global_patch()\n    try:\n        from vllm.v1.executor import multiproc_executor"
if a in s:
    assert s.count(a) == 1
    s = s.replace(a, b)
elif b not in s:
    raise SystemExit("Unexpected worker hook; no files changed.")
a = "        patch_vllm_dspark_draft_load_config()\n        try:\n            instance = super().__new__(cls, **kwargs)"
b = "        from verl_speco.integration.opd_vllm_tokens import patch_v2_draft_token_copy\n\n        patch_v2_draft_token_copy()\n" + a
if b not in s:
    assert s.count(a) == 1, "Unexpected extension; no files changed."
    s = s.replace(a, b)
a = '''        snapshot = (rid, request, len(request._output_token_ids), drafts, generated)
        if capture_draft_tokens and generated:
            tokens = list(scheduled.scheduled_spec_decode_tokens.get(rid, ()))[:drafts]
            if any(not isinstance(token, int) or token < 0 for token in tokens):
                raise ValueError("Draft-OPD replay requires concrete valid draft token IDs")
            snapshot += (tokens,)
        snapshots.append(snapshot)'''
b = '''        tokens = None
        if capture_draft_tokens and generated:
            tokens = list(scheduled.scheduled_spec_decode_tokens.get(rid, ()))[:drafts]
            admitted = any(
                item.req_id == rid for item in getattr(scheduled, "scheduled_new_reqs", ())
            )
            # Scheduler.schedule pads newly admitted/resumed decode requests
            # with -1 for uniform CUDA graphs. They have no actual proposal.
            if admitted and generated == 1 and tokens and all(t == -1 for t in tokens):
                drafts, tokens = 0, []
            if any(not isinstance(token, int) or token < 0 for token in tokens):
                raise ValueError(
                    "Draft-OPD replay requires concrete valid draft token IDs; "
                    f"request={rid!r}, generated={generated}, drafts={drafts}, "
                    f"admitted={admitted}, tokens={tokens!r}. "
                    "Check CUDA worker log for 'V2 real proposal CPU copy installed'."
                )
        snapshot = (rid, request, len(request._output_token_ids), drafts, generated)
        if tokens is not None:
            snapshot += (tokens,)
        snapshots.append(snapshot)'''
if a in t:
    assert t.count(a) == 1
    t = t.replace(a, b)
elif b not in t:
    raise SystemExit("Unexpected full replay trace version; no files changed.")
for content in (s, t):
    ast.parse(content)
for path, content in ((runtime, s), (trace, t)):
    if path.read_text(encoding="utf-8") == content:
        continue
    backup = path.with_name(path.name + f".before_cuda_replay_fix.{time.time_ns()}")
    backup.write_bytes(path.read_bytes())
    path.write_text(content, encoding="utf-8", newline="\n")
    print("Updated:", path, "Backup:", backup)
print("CUDA replay hotfix installed; restart this job and its rollout workers.")

PY
