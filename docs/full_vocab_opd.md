# Small-scale full-vocabulary OPD

`verl_speco.integration.full_vocab_opd.score_teacher_hidden` accepts an AsyncLLM,
context token IDs, sampled response token IDs and a unique request ID. The engine
must use CUDA Model Runner V2, synchronous scheduling and hidden capture enabled.
The function scores the concatenated token sequence with one extra generated token
and returns the final hidden rows that predict the response. The extra token is
ignored. Use token IDs from the same tokenizer and model revision as the teacher.

`reverse_kl_from_hidden` computes mean KL(student || teacher) for already selected
valid response rows. Pass aligned two-dimensional hidden tensors and output-head
weights on the same device, plus the teacher result's logits_contract. Teacher
inputs are detached from gradients. Apply padding and loss masks before this call.

This initial implementation supports standard unquantized bias-free heads, with
an unscaled student head; teacher scale, soft-cap and vocabulary truncation follow
the returned contract. Qwen3-1.7B is the validated teacher/student smoke model.
Custom logits computation, head precision overrides, quantized heads and vocabulary
remapping need separate validation. Use the exact teacher checkpoint for its head.

Projection is chunked by token rows (default 32). Student autograd intermediates
remain live until backward; this is not a fused or memory-bounded training kernel.
Hidden delivery still uses the existing IPC payload and capture admission budget.
Keep sequences and concurrency small until streaming store delivery is available.
These functions do not change the existing top-k OPD training configuration or
launch a training job automatically.

## Teacher data delivery without training

`score_teacher_payload` scores supplied `prompt_ids + response_ids` and returns
row-aligned data. It does not sample a student response, run a loss, or update
parameters. The capture uses absolute positions
`[len(prompt_ids)-1, len(prompt_ids)+len(response_ids)-1)`.

The payload carries `teacher_model` (the checkpoint identifier supplied by the
caller), prompt token IDs, segments, hidden layout/layer metadata and the vLLM
`logits_contract`. Its `tensors` contain the captured hidden states, absolute
`hidden_positions` and the corresponding `response_token_ids`. Consumers can
load the matching teacher output head from that checkpoint; output-head weights
and full-vocabulary logits are not transferred for every request.

Without TransferQueue, these tensors remain inline. With the existing SpeCo
TransferQueue configuration enabled, tensors are published in row chunks and
replaced by `tensor_chunks` references. Each reference records a key, row range,
field shapes/dtypes and checksums. Metadata stays small and JSON-serializable.
The same interface works with TQ SimpleStorage and MooncakeStore. This reuses
`live_hidden_transport` and `transferqueue_bridge`; it does not change vLLM.

```python
from verl_speco.integration.full_vocab_opd import score_teacher_payload
from verl_speco.integration.live_hidden_transport import (
    iter_tensor_chunks, release_tensor_chunks,
)

# config is the existing SpeCo transport configuration containing transfer_queue.
# Start its task-level TQ owner before scoring; keep it alive through all readers.
payload = await score_teacher_payload(
    teacher_engine, prompt_ids, response_ids, request_id,
    teacher_model="/data/model/Qwen3-1.7B",
    transport_config=config, chunk_rows=256,
)
try:
    for tensors in iter_tensor_chunks(payload, config):
        consume(tensors, payload["logits_contract"], payload["teacher_model"])
finally:
    # The owner does this only after ALL consumers finish or fail.
    release_tensor_chunks(payload)
```

Use a unique `transfer_queue.run_id`; use a teacher-specific `partition_id`, such
as `speco_teacher_features`. Each process uses one transfer configuration. The
owner starts the shared controller; consumers attach and close their local
clients. A reader does not delete chunks. Publication failure clears attempted
chunks, and the owner cleans up after consumer errors/cancellation. Do not stop
the owner/master while readers are active. Mooncake still requires its usual
shared service/network configuration across hosts.

**Boundary:** this is chunked delivery from verl-SpeCo after vLLM returns the
capture. The consumer can process chunks without concatenating the full result,
but vLLM still buffers the capture internally and sends it across its existing
IPC path. This does not remove `max_hidden_capture_bytes`, IPC limits, or vLLM's
whole-window residency. Publishing directly from intermediate prefill chunks
would require a vLLM connector/store change and is outside this implementation.

A small check is provided in `examples/check_teacher_payload_delivery.py`.
It scores eight teacher rows, writes three chunks and verifies two separate
reader processes before releasing the references. It runs zero training steps:

```bash
PYTHONPATH=$PWD VLLM_USE_V2_MODEL_RUNNER=1 OMP_NUM_THREADS=2 \
  /data/logprob-validation/.venv/bin/python examples/check_teacher_payload_delivery.py \
  --backend MooncakeStore --model /data/model/Qwen3-1.7B --output /data/teacher-delivery.json
```

Add `--synthetic` for a CPU-only store check, or use `--backend SimpleStorage`.
The check starts its own small Mooncake master when requested and cleans up its
resources. It does not patch the inference engine or invoke an optimizer.

### Delivery sizing and client lifetime

When `chunk_rows` is omitted, tensor delivery uses chunks of approximately
16 MiB of raw tensor data. A single row wider than this limit is still sent as
one row. Explicit `chunk_rows` preserves fixed-row delivery. This avoids repeated
synchronous store calls for small teacher payloads while bounding large reads.
The vLLM whole-window buffering boundary is unchanged.

Keep the TQ client alive for a consumer process or actor and close it at teardown.
With TQ 0.1.10, treat close as terminal for a Mooncake consumer: restarting it
requires a new worker process. Clearing the closed singleton alone is insufficient
because real Mooncake reconnect tests also lost backing storage rows. All readers
and reference cleanup must finish before client closure.

The vLLM logits contract currently covers unquantized Qwen3 and includes the
checkpoint head weight name. A missing contract means the teacher payload helper
cannot promise logits reconstruction, even though raw hidden capture still works.
