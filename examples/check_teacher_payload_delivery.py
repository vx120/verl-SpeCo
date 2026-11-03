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
from verl_speco.integration.full_vocab_opd import (
    score_rollout_teacher_payloads,
    score_teacher_payload,
)
from verl_speco.integration.live_hidden_transport import (
    fingerprint,
    iter_tensor_chunks,
    put_tensor_chunks,
    release_tensor_chunks,
)


@ray.remote(num_cpus=1)
class PayloadReader:
    def read(self, payload, config):
        chunks = list(iter_tensor_chunks(payload, config))
        return {
            name: fingerprint(torch.cat([chunk[name] for chunk in chunks]))
            for name in chunks[0]
        }

    def close(self):
        bridge.close_transfer_queue_client()


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
    payloads = []
    readers = []
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
            payloads = payload if isinstance(payload, list) else [payload]
            payload = None
            readers = [PayloadReader.remote() for _ in range(2)]
            samples = []
            for item in payloads:
                json.dumps(item)
                assert "tensors" not in item
                chunks = list(iter_tensor_chunks(item, config))
                expected = {
                    name: fingerprint(torch.cat([c[name] for c in chunks]))
                    for name in chunks[0]
                }
                for _ in range(2):
                    results = ray.get(
                        [reader.read.remote(item, config) for reader in readers],
                        timeout=45,
                    )
                    assert results == [expected, expected]
                samples.append(
                    {
                        "request_id": item.get("request_id"),
                        "rows": item["tensor_rows"],
                        "chunks": len(item["tensor_chunks"]),
                        "fields": list(expected),
                        "logits_contract": item.get("logits_contract"),
                    }
                )
            keys = {r["key"] for item in payloads for r in item["tensor_chunks"]}
            while payloads:
                release_tensor_chunks(payloads[-1])
                payloads.pop()
            assert not keys.intersection(bridge.list_samples())
            report = {
                "backend": args.backend,
                "synthetic": args.synthetic,
                "dataset": str(args.dataset) if args.dataset else None,
                "student_model": args.student_model if args.dataset else None,
                "teacher_model": args.model,
                "samples": samples,
                "readers": 2,
                "reads_per_reader": 2,
                "exact": True,
                "released": True,
                "training_steps": 0,
            }
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(report))
        finally:
            if payload is not None and not isinstance(payload, list):
                release_tensor_chunks(payload)
            for item in payloads:
                release_tensor_chunks(item)
            if readers:
                ray.get([reader.close.remote() for reader in readers], timeout=15)
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

    rollout = await generate_rollout_batch(args) if args.dataset else None
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            dtype="bfloat16",
            max_model_len=4096 if args.dataset else 64,
            max_num_seqs=args.samples if args.dataset else 1,
            gpu_memory_utilization=0.35,
            enforce_eager=True,
            async_scheduling=False,
            aux_output_config={"enable_hidden_states": True},
        )
    )
    try:
        if rollout is not None:
            return await score_rollout_teacher_payloads(
                engine,
                rollout,
                run_id,
                teacher_model=args.model,
                transport_config=config,
            )
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


async def generate_rollout_batch(args):
    from itertools import islice

    from verl import DataProto
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    from verl_speco.integration.vllm_runtime import (
        patch_transformers_attention_layer_type_constants,
    )
    from verl_speco.producer.input_reader import iter_input_records

    patch_transformers_attention_layer_type_constants()
    student = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.student_model,
            dtype="bfloat16",
            max_model_len=4096,
            max_num_seqs=args.samples,
            gpu_memory_utilization=0.35,
            enforce_eager=True,
            async_scheduling=False,
        )
    )
    try:
        tokenizer = student.get_tokenizer()
        prompts, responses = [], []
        for record in islice(
            iter_input_records(args.dataset, on_error="raise"), args.samples
        ):
            prompt = (
                tokenizer.encode(record.prompt)
                if isinstance(record.prompt, str)
                else tokenizer.encode(
                    tokenizer.apply_chat_template(
                        list(record.prompt),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    ),
                    add_special_tokens=False,
                )
            )
            if len(prompt) + args.max_tokens > 4096:
                raise ValueError(
                    "Dataset prompt exceeds this short-check context limit"
                )
            async for output in student.generate(
                {"prompt_token_ids": prompt},
                SamplingParams(
                    temperature=0,
                    max_tokens=args.max_tokens,
                    output_kind=RequestOutputKind.FINAL_ONLY,
                ),
                f"student-{record.sample_id}",
            ):
                if output.finished:
                    prompts.append(prompt)
                    responses.append(list(output.outputs[0].token_ids))
        if not prompts:
            raise ValueError("Dataset has no samples")
        prompt_width = max(map(len, prompts))
        response_width = max(map(len, responses))
        prompt_batch = torch.full(
            (len(prompts), prompt_width), tokenizer.pad_token_id, dtype=torch.long
        )
        response_batch = torch.full(
            (len(prompts), response_width), tokenizer.pad_token_id, dtype=torch.long
        )
        attention = torch.zeros(
            len(prompts), prompt_width + response_width, dtype=torch.long
        )
        for i, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
            prompt_batch[i, -len(prompt) :] = torch.tensor(prompt)
            response_batch[i, : len(response)] = torch.tensor(response)
            attention[i, prompt_width - len(prompt) : prompt_width + len(response)] = 1
        return DataProto.from_dict(
            tensors={
                "prompts": prompt_batch,
                "responses": response_batch,
                "attention_mask": attention,
                "response_mask": attention[:, prompt_width:].clone(),
            }
        )
    finally:
        student.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["SimpleStorage", "MooncakeStore"], default="MooncakeStore"
    )
    parser.add_argument("--model", default="/data/model/Qwen3-1.7B")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--student-model", default="/data/model/Qwen3-1.7B")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
