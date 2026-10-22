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
logger.setLevel(logging.INFO)


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
        remaining = set(bridge.list_samples()).intersection(keys)
        if remaining:
            raise RuntimeError(
                f"TQ keys remain after collection release: {sorted(remaining)}"
            )
        logger.info("[live TQ release] count=%s remaining=0 keys=%s", len(keys), keys)
