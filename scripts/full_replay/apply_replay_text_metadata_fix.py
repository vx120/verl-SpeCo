from pathlib import Path
import ast
import sys
import time

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/model/ljh/draaft-opd_VAT/verl-SpeCo/verl_speco")
p = root / "integration/opd_replay.py"
s = p.read_text(encoding="utf-8")
a = '''    if (
        "multi_modal_inputs" in batch.non_tensor_batch
        or tensors["position_ids"].ndim != 2
    ):
        raise ValueError(
            "Draft-OPD branch replay currently supports text-only position IDs"
        )'''
b = '''    # verl's text-only agent loop also carries a per-row multimodal column
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
        )'''
if a in s:
    assert s.count(a) == 1, "Unexpected replay source; no files changed."
    fixed = s.replace(a, b)
    ast.parse(fixed)
    backup = p.with_name(p.name + f".before_text_metadata_fix.{time.time_ns()}")
    backup.write_bytes(p.read_bytes())
    p.write_text(fixed, encoding="utf-8", newline="\n")
    print("Updated:", p, "Backup:", backup)
elif b in s:
    print("Already patched:", p)
else:
    raise SystemExit("Unexpected replay version; no files changed.")
