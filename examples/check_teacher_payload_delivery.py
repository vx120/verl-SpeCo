"""Check teacher payload delivery with two readers; no training or optimizer."""

import argparse
import asyncio
import json
import socket
import subprocess
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

import ray
import torch
import transfer_queue as tq

from verl_speco.integration import transferqueue_bridge as bridge
from verl_speco.integration.full_vocab_opd import score_teacher_payload
from verl_speco.integration.live_hidden_transport import (
    fingerprint,
    iter_tensor_chunks,
    put_tensor_chunks,
    release_tensor_chunks,
)


@ray.remote(num_cpus=1)
def read_payload(payload, config):
    from verl_speco.integration import transferqueue_bridge as client
    from verl_speco.integration.live_hidden_transport import (
        fingerprint,
        iter_tensor_chunks,
    )

    try:
        chunks = list(iter_tensor_chunks(payload, config))
        return {
            name: fingerprint(torch.cat([chunk[name] for chunk in chunks]))
            for name in chunks[0]
        }
    finally:
        client.close_transfer_queue_client()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main(args):
    run_id = f"teacher-delivery-{uuid.uuid4().hex}"
    config = {
        "transfer_queue": {
            "enable": True,
            "run_id": run_id,
            "partition_id": "speco_teacher_features",
            "controller": {"polling_mode": True},
            "backend": {"storage_backend": args.backend},
        }
    }
    master = payload = None
    with ExitStack() as resources:
        try:
            if args.backend == "MooncakeStore":
                port = free_port()
                master_log = resources.enter_context(
                    open(str(args.output) + ".master.log", "w")
                )
                master = subprocess.Popen(
                    [
                        "mooncake_master",
                        "--rpc_address=127.0.0.1",
                        f"--rpc_port={port}",
                        "--default_kv_lease_ttl=120",
                        "--default_kv_soft_pin_ttl=120",
                    ],
                    stdout=master_log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 15
                while True:
                    if master.poll() is not None:
                        raise RuntimeError("Mooncake master exited during startup")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                            break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Mooncake master startup timed out"
                            ) from None
                        time.sleep(0.1)
                config["transfer_queue"]["backend"]["MooncakeStore"] = {
                    "auto_init": False,
                    "metadata_server": "P2PHANDSHAKE",
                    "master_server_address": f"127.0.0.1:{port}",
                    "local_hostname": "127.0.0.1",
                    "protocol": "tcp",
                    "global_segment_size": 64 * 1024**2,
                    "local_buffer_size": 16 * 1024**2,
                    "use_gdr": False,
                }
            else:
                config["transfer_queue"]["backend"]["SimpleStorage"] = {
                    "total_storage_size": 100,
                    "num_data_storage_units": 1,
                }
            ray.init(
                num_cpus=4,
                include_dashboard=False,
                namespace=run_id,
                object_store_memory=80 * 1024**2,
            )
            bridge.start_transfer_queue_owner(config)
            if args.synthetic:
                payload = put_tensor_chunks(
                    {
                        "global_step": 0,
                        "replica_rank": 0,
                        "collection_id": run_id,
                        "tensors": {
                            "hidden_states": torch.arange(32)
                            .reshape(8, 4)
                            .to(torch.bfloat16),
                            "hidden_positions": torch.arange(3, 11),
                            "response_token_ids": torch.arange(500, 508),
                        },
                    },
                    config,
                    chunk_rows=3,
                )
            else:
                payload = asyncio.run(capture_teacher(args, config, run_id))
            json.dumps(payload)
            assert "tensors" not in payload and len(payload["tensor_chunks"]) == 3
            expected = {
                name: fingerprint(
                    torch.cat([c[name] for c in iter_tensor_chunks(payload, config)])
                )
                for name in ("hidden_states", "hidden_positions", "response_token_ids")
            }
            readers = [read_payload.remote(payload, config) for _ in range(2)]
            results = ray.get(readers, timeout=45)
            assert results == [expected, expected]
            release_tensor_chunks(payload)
            keys = {r["key"] for r in payload["tensor_chunks"]}
            assert not keys.intersection(bridge.list_samples())
            payload = None
            report = {
                "backend": args.backend,
                "synthetic": args.synthetic,
                "chunks": 3,
                "rows": 8,
                "readers": 2,
                "exact": True,
                "released": True,
                "training_steps": 0,
            }
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(report))
        finally:
            if payload is not None:
                release_tensor_chunks(payload)
            client = tq.get_client()
            if client is not None:
                client.close()
            bridge.close_transfer_queue_owner()
            ray.shutdown()
            if master is not None:
                master.terminate()
                try:
                    master.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    master.kill()
                    master.wait()


async def capture_teacher(args, config, run_id):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            dtype="bfloat16",
            max_model_len=64,
            max_num_seqs=1,
            gpu_memory_utilization=0.35,
            enforce_eager=True,
            async_scheduling=False,
            aux_output_config={"enable_hidden_states": True},
        )
    )
    try:
        return await score_teacher_payload(
            engine,
            [100, 200, 300, 400],
            list(range(500, 508)),
            run_id,
            teacher_model=args.model,
            transport_config=config,
            chunk_rows=3,
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["SimpleStorage", "MooncakeStore"], default="MooncakeStore"
    )
    parser.add_argument("--model", default="/data/model/Qwen3-1.7B")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
