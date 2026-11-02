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

"""Online CPU hidden-state transport through TransferQueue, with no file staging.

Small routing/position metadata still follows the existing Ray control path.
The TaskRunner owns the queue; sample keys are cleared after all collection
RPCs complete, rather than by individual FSDP readers.
"""

import hashlib
import logging
import time

logger = logging.getLogger(__name__)


def fingerprint(tensor):
    import torch

    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def put_hidden(sample, training):
    cfg = training.get("transfer_queue", {})
    if not cfg.get("enable", False):
        return sample
    from verl_speco.integration import transferqueue_bridge as bridge

    if not bridge.configure_transfer_queue(training):
        raise RuntimeError("Requested online TQ transport is unavailable")
    tensor = sample["hidden_states"].detach().cpu().contiguous()
    run_id = cfg.get("run_id")
    if not run_id:
        raise ValueError("Online TQ requires a unique run_id")
    key = f"{run_id}:{sample['global_step']}:{sample['replica_rank']}:{sample['collection_id']}"
    started = time.perf_counter()
    checksum = fingerprint(tensor)
    bridge.put_sample(
        key,
        {"hidden_states": tensor},
        tag={
            "step": int(sample["global_step"]),
            "owner": int(sample["replica_rank"]),
            "collection_id": str(sample["collection_id"]),
        },
    )
    result = dict(sample)
    del result["hidden_states"]
    result["hidden_states_tq_ref"] = {
        "key": key,
        "sha256": checksum,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }
    logger.info(
        "[live TQ put] key=%s step=%s owner=%s bytes=%s elapsed=%.6f backend=%s",
        key,
        sample["global_step"],
        sample["replica_rank"],
        tensor.numel() * tensor.element_size(),
        time.perf_counter() - started,
        cfg["backend"]["storage_backend"],
    )
    return result


def get_hidden(sample, training):
    ref = sample.get("hidden_states_tq_ref")
    if ref is None:
        return sample.get("hidden_states")
    from verl_speco.integration import transferqueue_bridge as bridge

    if not bridge.configure_transfer_queue(training):
        raise RuntimeError("TQ reference received without active TQ configuration")
    started = time.perf_counter()
    # kv_batch_get returns a TensorDict batch even for one key. The batched
    # bridge API explicitly splits that transport dimension before returning.
    tensor = bridge.get_samples([ref["key"]])[0][1]["hidden_states"]
    if list(tensor.shape) != ref["shape"] or str(tensor.dtype) != ref["dtype"]:
        raise ValueError(
            f"Online TQ hidden shape/dtype mismatch: {tuple(tensor.shape)}, {tensor.dtype}; expected {ref['shape']}, {ref['dtype']}"
        )
    if fingerprint(tensor) != ref["sha256"]:
        raise ValueError("Online TQ hidden checksum mismatch")
    logger.info(
        "[live TQ get] key=%s step=%s owner=%s exact=1 elapsed=%.6f",
        ref["key"],
        sample["global_step"],
        sample["replica_rank"],
        time.perf_counter() - started,
    )
    return tensor


def release_collection(payload):
    keys = sorted(
        {
            s["hidden_states_tq_ref"]["key"]
            for bucket in payload.buckets
            for s in bucket
            if "hidden_states_tq_ref" in s
        }
    )
    if keys:
        from verl_speco.integration import transferqueue_bridge as bridge

        bridge.clear_samples(keys)
        logger.info("[live TQ release] count=%s keys=%s", len(keys), keys)


def put_tensor_chunks(sample, training, *, chunk_rows=256):
    """Publish row-aligned tensors; return metadata and ordered store references."""
    cfg = training.get("transfer_queue", {})
    if not cfg.get("enable", False):
        return sample
    from verl_speco.integration import transferqueue_bridge as bridge

    if not bridge.configure_transfer_queue(training):
        raise RuntimeError("Requested tensor transport is unavailable")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    run_id = cfg.get("run_id")
    if not run_id:
        raise ValueError("Tensor transport requires a unique run_id")
    tensors = sample["tensors"]
    rows = next(iter(tensors.values())).shape[0]
    if rows < 1 or any(t.shape[0] != rows for t in tensors.values()):
        raise ValueError("Payload tensors must have the same nonempty row dimension")
    prefix = f"{run_id}:tensors:{sample['global_step']}:{sample['replica_rank']}:{sample['collection_id']}"
    references, attempted_keys = [], []
    try:
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            key = f"{prefix}:{start}"
            fields = {
                name: tensor[start:end].detach().cpu().contiguous()
                for name, tensor in tensors.items()
            }
            metadata = {
                name: {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "sha256": fingerprint(t),
                }
                for name, t in fields.items()
            }
            attempted_keys.append(key)
            bridge.put_sample(key, fields, tag={"start": start, "end": end})
            references.append(
                {"key": key, "start": start, "end": end, "tensors": metadata}
            )
    except Exception:
        bridge.clear_samples(attempted_keys)
        raise
    result = {name: value for name, value in sample.items() if name != "tensors"}
    result.update(tensor_rows=rows, tensor_chunks=references)
    return result


def iter_tensor_chunks(sample, training):
    """Read one chunk at a time without consuming references needed by other readers."""
    if "tensor_chunks" not in sample:
        yield sample["tensors"]
        return
    from verl_speco.integration import transferqueue_bridge as bridge

    if not bridge.configure_transfer_queue(training):
        raise RuntimeError(
            "Tensor references require active TransferQueue configuration"
        )
    cursor = 0
    for ref in sample["tensor_chunks"]:
        if ref["start"] != cursor or ref["end"] <= cursor:
            raise ValueError("Tensor chunk row ranges are not contiguous")
        fields = bridge.get_samples([ref["key"]])[0][1]
        if set(fields) != set(ref["tensors"]):
            raise ValueError("Tensor chunk fields do not match the manifest")
        for name, tensor in fields.items():
            expected = ref["tensors"][name]
            if (
                list(tensor.shape) != expected["shape"]
                or tensor.shape[0] != ref["end"] - ref["start"]
                or str(tensor.dtype) != expected["dtype"]
                or fingerprint(tensor) != expected["sha256"]
            ):
                raise ValueError(f"Tensor chunk integrity mismatch: {name}")
        cursor = ref["end"]
        yield fields
    if cursor != sample["tensor_rows"]:
        raise ValueError("Tensor chunk manifest is incomplete")


def release_tensor_chunks(sample):
    """Owner cleanup after all consumers have finished, including failed consumption."""
    keys = [ref["key"] for ref in sample.get("tensor_chunks", [])]
    if keys:
        from verl_speco.integration import transferqueue_bridge as bridge

        bridge.clear_samples(keys)
