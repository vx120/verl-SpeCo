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

"""Native two-step independent training; TQ/Mooncake consumed by torchrun directly."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def tensor_digest(tensor):
    import torch

    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(value)).hexdigest()


def samples_from_raw(raw):
    import torch
    from types import SimpleNamespace
    from verl_speco.trainer.target_feature_replay import (
        FeatureContract,
        feature_from_vllm_payload,
    )

    contract = FeatureContract(
        algorithm="DSPARK",
        target_layer_ids=[1, 9, 17, 25, 33],
        vllm_aux_hidden_state_layer_ids=[2, 10, 18, 26, 34],
        hidden_states_layout="dflash_aux_plus_last",
        dtype=torch.bfloat16,
        target_model_id=os.environ["TARGET_MODEL"],
        target_model_revision=None,
        tokenizer_fingerprint="local-Qwen3-4B",
        target_num_hidden_layers=36,
        require_full_alignment=True,
    )
    samples = []
    for prompt, generated, payload in zip(
        raw["prompts"], raw["generated"], raw["payloads"]
    ):
        ids = torch.tensor(prompt["prompt_token_ids"] + generated[:-1])
        positions = payload["hidden_positions"]
        mask = torch.zeros(len(ids))
        mask[len(prompt["prompt_token_ids"]) :] = 1
        request = SimpleNamespace(
            input_ids=ids,
            loss_mask=mask,
            feature_positions=positions,
            draft_position_ids=positions + 1,
            source_metadata={},
        )
        samples.append(feature_from_vllm_payload(payload, request, contract))
    assert len(samples) == 2
    return samples


def train_command(root, transport, tq_config=None, input_path=None):
    prefix = "actor_rollout_ref.rollout.drafter."
    training = prefix + "training."
    overrides = [
        "speco.draft_training.runtime_backend=subprocess",
        "speco.draft_training.nproc_per_node=1",
        "actor_rollout_ref.model.path=" + os.environ["TARGET_MODEL"],
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        prefix + "model_path=" + os.environ["DRAFT_MODEL"],
        prefix + "speculative_algorithm=DSPARK",
        prefix + f"checkpoint_path={root}/checkpoints",
        training
        + "feature_store.type="
        + ("tq" if transport == "mooncake" else "torch_shard"),
        training + f"feature_store.path={root}/shards",
        training + "feature_store.shuffle=false",
        training + "max_steps=2",
        training + "save_final_checkpoint=true",
        training + "save_interval_steps=0",
        training + "batch_size_per_gpu=1",
        training + "dspark_num_anchors=2",
        training + "dspark_max_window=32",
        training + "dspark_loss_mode=restricted_ce",
        training + "dspark_l1_loss_alpha=0.45",
        training + "dspark_distribution_loss_impl=eager",
        training + "seed=42",
        "trainer.logger=[console]",
        f"trainer.default_local_dir={root}/trainer-output",
    ]
    if input_path is not None:
        overrides.append("data.train_files=" + str(input_path))
    if tq_config:

        def visit(mapping, parent):
            for key, value in mapping.items():
                full = parent + "." + key
                if isinstance(value, dict):
                    yield from visit(value, full)
                else:
                    rendered = (
                        json.dumps(value) if not isinstance(value, str) else value
                    )
                    required = {
                        training + "transfer_queue.enable",
                        training + "transfer_queue.run_id",
                        training + "transfer_queue.ray.address",
                    }
                    yield ("" if full in required else "++") + full + "=" + rendered

        overrides.extend(visit(tq_config, training + "transfer_queue"))
    return [sys.executable, "-m", "verl_speco.draft_train_launcher", *overrides]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=["ordinary", "mooncake"], required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    args = parser.parse_args()
    import torch

    samples = samples_from_raw(
        torch.load(args.raw, map_location="cpu", weights_only=False)
    )
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    records = [
        {
            "hidden_sha256": tensor_digest(s.hidden_states),
            "shape": list(s.hidden_states.shape),
        }
        for s in samples
    ]
    (root / "input-features.json").write_text(json.dumps(records, indent=2))
    if args.transport == "ordinary":
        from verl_speco.trainer.feature_store import TorchShardFeatureStore

        store = TorchShardFeatureStore(root / "shards", max_samples_per_shard=2)
        store.write_many(samples)
        store.flush()
        if args.cpu_preflight:
            print("ORDINARY_PREPARE_PASS", flush=True)
            return
        command = train_command(root, args.transport, input_path=args.raw)
        (root / "train-command.json").write_text(json.dumps(command, indent=2))
        with (root / "train.log").open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=600
            )
        (root / "exit.txt").write_text(str(result.returncode))
        assert result.returncode == 0
        return

    import ray
    from verl_speco.integration import transferqueue_bridge as bridge
    from verl_speco.transport.drafter_sample_protocol import (
        SampleMetadata,
        encode_sample,
        make_ready_tag,
        make_sample_key,
        PROTOCOL_SCHEMA_VERSION,
    )
    from verl_speco.trainer.tq_feature_store import TQFeatureStore
    from verl_speco.tq_owner import publish_owner_ready

    run_id = root.name
    ray_context = ray.init(
        num_cpus=4, include_dashboard=False, namespace=run_id, log_to_driver=True
    )
    address = ray_context.address_info["gcs_address"]
    cfg = {
        "enable": True,
        "run_id": run_id,
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "partition_id": "speco_drafter_features",
        "ray": {"address": address, "namespace": run_id},
        "controller": {"polling_mode": True},
        "drop_last": True,
        "backend": {
            "storage_backend": "MooncakeStore",
            "MooncakeStore": {
                "auto_init": False,
                "metadata_server": "P2PHANDSHAKE",
                "master_server_address": os.environ["MOONCAKE_MASTER_ADDRESS"],
                "local_hostname": os.environ.get(
                    "MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"
                ),
                "protocol": "tcp",
                "global_segment_size": 67108864,
                "local_buffer_size": 16777216,
                "use_gdr": False,
            },
        },
    }
    consumer = None
    try:
        bridge.start_transfer_queue_owner(cfg)
        publish_owner_ready(run_id, PROTOCOL_SCHEMA_VERSION)
        keys = []
        for i, sample in enumerate(samples):
            meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, run_id, f"sample-{i}", i)
            key = make_sample_key(meta)
            fields = encode_sample(sample, meta)
            bridge.put_sample(key, fields, tag=make_ready_tag(meta))
            keys.append(key)
            print("INDEPENDENT_TQ_PUT", key, records[i]["hidden_sha256"], flush=True)

        # Actual cross-process decode/checksum preflight, then native Trainer
        # consumes and clears the same two records after optimizer updates.
        @ray.remote(num_cpus=1)
        class Reader:
            def read(self, config):
                store = TQFeatureStore(config)
                store.connect()
                entries = store.list_ready()
                values = store.get_many(entries)
                hashes = [tensor_digest(s.hidden_states) for s in values]
                store.close_local()
                return hashes

        reader = Reader.remote()
        hashes = ray.get(reader.read.remote(cfg), timeout=90)
        assert hashes == [r["hidden_sha256"] for r in records]
        ray.kill(reader)
        print("INDEPENDENT_TQ_EXACT_PASS", json.dumps(hashes), flush=True)
        if args.cpu_preflight:
            bridge.clear_samples(keys)
        else:
            command = train_command(root, args.transport, cfg, input_path=args.raw)
            (root / "train-command.json").write_text(json.dumps(command, indent=2))
            with (root / "train.log").open("w") as log:
                consumer = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    result = consumer.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    consumer.terminate()
                    consumer.wait(timeout=30)
                    raise
            (root / "exit.txt").write_text(str(result))
            assert result == 0
        remaining = set(bridge.list_samples()).intersection(keys)
        assert not remaining, remaining
        (root / "transport.json").write_text(
            json.dumps(
                {
                    "backend": "MooncakeStore",
                    "protocol": "cpu_tcp",
                    "keys": keys,
                    "exact_hidden_hashes": hashes,
                    "remaining_sample_keys": list(remaining),
                    "native_tq_consumer": not args.cpu_preflight,
                    "input_file": str(args.raw),
                    "scope": "Two-stage Independent; native Trainer reads TQ directly, no intermediate Mooncake result file/TorchShard",
                },
                indent=2,
            )
        )
        print("INDEPENDENT_TQ_RELEASE_PASS", flush=True)
    finally:
        if consumer and consumer.poll() is None:
            consumer.terminate()
            consumer.wait(timeout=30)
        bridge.close_transfer_queue_owner()
        ray.shutdown()


if __name__ == "__main__":
    main()
