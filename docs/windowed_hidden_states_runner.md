# Runner windowed hidden-state collection

Last updated: 10/22/2026

This branch integrates the Runner capture implementation in
[`vx120/vllm`, `windowed-hidden-state-collection`](https://github.com/vx120/vllm/tree/windowed-hidden-state-collection)
with SpeCo. The tested vLLM commit is
`df01e5f1f19003fb803162364eb8438e39bf6bc1`; the SpeCo base is
`f868e84a8acc768c95ec6e764dfc3a8ec67ab0a1`.

## Scope

The implemented Co-Train path is:

```text
real DSpark speculative rollout
  -> Runner window capture
  -> position-aware drafter sample
  -> inline CPU tensor or TQ/Mooncake CPU tensor reference
  -> native drafter worker update
  -> draft weight publication to rollout replicas
```

The ordinary and live Mooncake paths each completed one two-step GPU smoke run
with Qwen3-4B and DSpark on two RTX 5090 GPUs. Each run collected four samples
per step, performed two native drafter updates, and committed revisions 1 and 2
on both rollout replicas. Actor loss/grad were zero in this small test: it proves
the pipeline, not useful policy learning or model quality.

Independent training also completed two native updates using TorchShard and a
separate live TQ/Mooncake consumer. The included Independent example starts from
already collected features; it does not replace every default generation/replay
Producer workflow.

The runtime modules here are taken from those verified deployments, with formatting,
license headers and a local `Any` annotation for the dynamic upstream configuration.
Server-only
data paths in the experiment's `draft_trainer.yaml` are deliberately excluded.
Example launchers expose paths through environment variables. No model weights,
training data, credentials, server logs, or experiment result archives are included.

## Configuration and feature contract

Enable `actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_vllm`
and disable the SGLang/old-logprob capture sources when testing this source in
isolation. The old-logprob source remains available; Runner capture does not
remove RL's old-logprob computation.

The request window uses response coordinates, with absolute row
`abs = prompt_len - 1 + response_index`. A row at `abs` corresponds to input
token `x[abs]` and predicts `x[abs + 1]`. Only accepted speculative rows contribute
to the final result. Requests must provide enough generated tokens for the
requested window; unavailable capture is reported rather than inventing rows.

For the tested DSpark checkpoint, decoder layer IDs `[1,9,17,25,33]` map to vLLM
auxiliary output IDs `[2,10,18,26,34]`. The tested training layout appends normalized
final features, `dflash_aux_plus_last`. The explicit-window converter checks
positions, next-token labels, layer order, shape and final normalization metadata;
it preserves the legacy converter for payloads without `hidden_positions`.

`vllm_hidden_capture.py` also contains EAGLE3/DFLASH layout adapters covered by
CPU contract tests. The GPU Co-Train claim above applies to DSpark only.

## Co-Train smoke example

Create an environment using the repository's normal `uv` setup. Install the
matching vLLM implementation and a supported upstream verl release. The test
environment used verl 0.8.0, TransferQueue 0.1.10 and its MooncakeStore dependencies;
no claim is made that arbitrary newer versions preserve these interfaces.

```bash
export TARGET_MODEL=/models/Qwen3-4B
export DRAFT_MODEL=/models/dspark_qwen3_4b_block7
export TRAIN_DATA=/datasets/train.parquet
export OUTPUT_DIR="$PWD/outputs/runner-window-smoke"
export PYTHON_BIN="$PWD/.venv/bin/python"

bash examples/windowed_hidden_states/run_qwen3-4b_drafter_dspark_vllm.sh
```

This example retains the tested two-step, two-GPU, response64/window32 configuration
with eager execution, TRITON_ATTN, and prefix caching/chunked prefill/async scheduling
disabled. Supply compatible data, CUDA tools and backend dependencies for the host.
Additional Hydra overrides may be passed at the end of the command. Output directories
should be unique; the example is a smoke configuration, not a production default.

For live CPU TQ/Mooncake, start a dedicated Mooncake master using the installed
version's command-line interface, then set:

```bash
export SPECO_HIDDEN_TRANSPORT=mooncake
export MOONCAKE_MASTER_ADDRESS=127.0.0.1:50161
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export SPECO_RUN_ID=runner-window-smoke-unique-id
bash examples/windowed_hidden_states/run_qwen3-4b_drafter_dspark_vllm.sh
```

The launcher uses `auto_init=false`, TCP and `use_gdr=false`. It does not start or
stop a shared master. Stop only the dedicated master that you own after the job.
This is CPU transport, not GPU zero-copy/RDMA. Small routing metadata still travels
through the existing Ray control path. Hidden tensors do not use intermediate
feature-file staging in this Co-Train path.

`live_hidden_transport.py` replaces a hidden tensor with a unique run/step/replica/
collection reference. Consumers verify shape, dtype and SHA256. TaskRunner releases
the keys after the collection dispatch finishes, so multiple readers can consume
the same sample. Requested transport fails if unavailable rather than silently
returning to inline transport. Failure-path cleanup beyond the exercised smoke
conditions still needs broader validation.

## Independent smoke example

`examples/windowed_hidden_states/independent.py` consumes a trusted local `torch.save`
bundle with `prompts`, `generated` and `payloads` for two samples. Each prompt has
`prompt_token_ids`; generated entries contain response token IDs; each payload
contains explicit-window hidden states, positions, token/target IDs, layer IDs and
final normalization metadata. It uses the tested DSpark/Qwen3-4B layer contract.
Because the input uses `torch.load(..., weights_only=False)`, only load a bundle
that you created or otherwise trust.

```bash
uv run --no-project .venv/bin/python examples/windowed_hidden_states/independent.py \
  --raw /features/runner-window.pt --output /results/independent-ordinary-new \
  --transport ordinary

# With TARGET_MODEL, DRAFT_MODEL and MOONCAKE_MASTER_ADDRESS already set:
uv run --no-project .venv/bin/python examples/windowed_hidden_states/independent.py \
  --raw /features/runner-window.pt --output /results/independent-mooncake-new \
  --transport mooncake
```

The output directory must not exist. Each run performs two native updates and
saves a complete final checkpoint, so reserve disk space for model and optimizer
state. `--cpu-preflight` checks feature preparation/transport without launching
training. For Mooncake, the native Trainer consumes TQ directly, with an exact
cross-process input check and an empty-key postcondition. The initial feature
file is still part of this two-stage Independent workflow.

## Validation and limits

CPU tests, including a contract check against the actual Runner source:

```bash
export SPECO_RUNNER_CAPTURE_SOURCE=/src/vllm/vllm/v1/hidden_state_capture.py
uv run --no-project .venv/bin/python -m pytest \
  tests/unit/test_vllm_hidden_capture.py \
  tests/unit/test_windowed_features.py \
  tests/unit/test_live_hidden_transport.py \
  tests/unit/test_cotrain_tied_bucket.py \
  tests/integration/test_agent_loop_runtime_contract.py -q
```

Without the source path, four sibling-Runner cases skip if the expected sibling
checkout is absent. These CPU tests do not substitute for GPU integration tests.

Delivery validation on 2026-10-22: 168 related CPU tests passed, including the
five files above and the collection adapter/strategy, drafter scheduler/state,
target feature replay and TQ producer/consumer suites. All repository pre-commit
hooks passed. The portable shell launcher passed `bash -n`; the Independent
ordinary example passed `--cpu-preflight` on the archived A feature input.
No new GPU training was run while preparing this branch. Runtime AST comparison
against the deployed source found only the local configuration annotation beyond
formatting and license comments.

Separate controlled experiments found identical inputs, actual anchors, complete
loss records, and all 63 model tensors at initialization and after both updates
for ordinary versus Mooncake transport of the same B-produced feature input.
This validates that controlled transport/model trajectory only. Full optimizer
state was not independently fingerprinted, and it does not resolve the deferred
Connector cross-slot numerical issue.

Uniform-warmup common-extraction measurements were one batch per mode: Window32
API wall time was 1.200613s for A and 1.130560s for B. Both reduced output-related
bytes by approximately half relative to Full64, without a corresponding reduction
in the measured worker allocator peak. This is not a steady-state A/B ranking or
proof of a Mooncake speedup. The common A measurement used extraction auxiliary
outputs, not its native best speculative path.

Current A capability checks reject async scheduling and context parallelism, among
other unsupported mappings. Only supported auxiliary outputs and response-side
windows are available; this is not arbitrary-layer/arbitrary-position extraction.
Cancellation, large concurrent workloads, parallel layouts and resource failure
paths need further testing before production adoption. The branch does not claim
those follow-up optimizations are implemented.
