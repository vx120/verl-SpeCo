# Connector windowed features: experimental B adaptation

Last updated: 10/22/2026

This branch contains the Connector-based SpeCo experiments, including ordinary
and live CPU TQ/Mooncake train-only Co-Train and two-stage Independent training.
The original cross-slot numerical issue remains deferred. Successful pipeline
execution must not be interpreted as passing that feature-correctness gate.

## Matching versions

The GPU experiments used vLLM commit
`c7f20141f58e37844fd3abb3785ed94f7c090e5e`, available on the fixed
[`codex/connector-feature-export-tested-20261022`](https://github.com/vx120/vllm/tree/codex/connector-feature-export-tested-20261022)
branch. SpeCo's base is `f868e84a8acc768c95ec6e764dfc3a8ec67ab0a1`.

The vLLM [`connector-feature-export`](https://github.com/vx120/vllm/tree/connector-feature-export)
branch currently has the later `e8f763c5ae4c0a768abff59bd27803448ede8ea7`
commit with upstream/review changes. It is preserved, not rolled back. The old
GPU results are not validation of that newer commit; use the fixed branch when
reproducing the reported experiment version.

## Implemented path and limits

```text
vLLM extract_hidden_states method (no real DSpark serving drafter)
  -> Connector Window export to safetensors
  -> position/token/layer checks and CPU tensor conversion
  -> inline tensor or CPU TQ/Mooncake reference
  -> native DSpark optimizer update
  -> complete training checkpoint (no inference draft publication)
```

`connector_train_only=true` is mandatory for this variant's publication hook.
The hook saves a checkpoint after a successful update and returns
`drafter/published=0`. It rejects attempts to use the extraction engine as an
online DSpark weight-publication target. `drafter.speculative_algorithm=DSPARK`
selects the training backend here; it does not make the extraction engine perform
DSpark speculative decoding. Runtime identity handling explicitly distinguishes
the two roles.

The current Connector implementation occupies the `extract_hidden_states`
speculative method. Real DSpark speculative rollout plus online publication has
not been implemented in this B variant. Ordinary and Mooncake configurations each
completed two native updates and saved checkpoints in two-step Co-Train smoke
runs, but they are not equivalent to A's real speculative Co-Train.

The tested configuration uses Qwen3-4B, two RTX 5090 GPUs, response64/window32,
auxiliary output IDs `[2,10,18,26,34]` plus normalized final, BF16, MRV1, eager
execution and TRITON_ATTN. Async scheduling, prefix caching and chunked prefill
are disabled; TP/PP/CP are 1. Actor loss/grad were zero in the small smoke runs,
so no useful policy-learning or final-quality claim is made.

## Feature contract

For response index `r`, the captured source row is `prompt_len - 1 + r`, which
predicts the following token. The online adapter checks explicit positions,
token IDs, next-token IDs, layer order, finite values and tensor shape. It converts
`[rows,layers,hidden]` to the native DSpark flattened aux-plus-final layout.

The shared Independent converter handles explicit positions and final-normalization
metadata, while legacy payloads continue through the old converter. File readers
check `.error` after acquiring the completion lock to reject failed exports.
These checks detect protocol/alignment errors; they do not replace a numerical
oracle for captured hidden states.

The underlying extraction still has full-sequence cache allocation. Window masks
reduce written/exported rows, not the allocated sequence range. A measured halving
of file bytes must not be described as halving GPU cache or whole-card memory.

## Co-Train smoke launcher

Use the normal `uv` environment setup, install the matching vLLM build and upstream
verl, and provide your own models/data. The experiment environment used verl 0.8.0
and TransferQueue 0.1.10; arbitrary dependency versions are not validated.

```bash
export TARGET_MODEL=/models/Qwen3-4B
export DRAFT_MODEL=/models/dspark_qwen3_4b_block7
export TRAIN_DATA=/datasets/train.parquet
export OUTPUT_DIR="$PWD/outputs/connector-smoke-new"
export PYTHON_BIN="$PWD/.venv/bin/python"
bash examples/connector_hidden_states/run_qwen3-4b_drafter_dspark_vllm.sh
```

This is a two-step, two-GPU configuration. Checkpoints include optimizer state;
reserve disk space and use a new output directory. The example parameterizes the
tested launch settings rather than embedding the original server paths. Additional
Hydra overrides can be appended. It has not been rerun as a new GPU experiment
while preparing this delivery.

For live CPU TQ/Mooncake, start a dedicated master and set:

```bash
export SPECO_HIDDEN_TRANSPORT=mooncake
export MOONCAKE_MASTER_ADDRESS=127.0.0.1:50171
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export SPECO_RUN_ID=connector-smoke-unique-id
bash examples/connector_hidden_states/run_qwen3-4b_drafter_dspark_vllm.sh
```

The launcher uses `auto_init=false`, CPU TCP and `use_gdr=false`. It does not manage
a shared master; stop only your dedicated process after the job. Exported files
still exist before tensor publication to TQ. This is not a file-free pipeline,
GPU zero-copy or RDMA validation.

Reference keys include run, step, replica and collection. The consumer validates
shape/dtype/SHA256; the collection owner releases keys after dispatch to all readers
and checks the release postcondition. There is no silent inline fallback when a
requested transport is unavailable. Broader failure-path cleanup remains unproven.

## Independent training

The native Independent ordinary and live Mooncake paths each completed two updates
and saved a complete final checkpoint. The example consumes an already collected,
trusted local `torch.save` feature bundle with two `prompts`, `generated` entries
and explicit-window `payloads` in the same DSpark/Qwen3-4B contract.

```bash
uv run --no-project .venv/bin/python examples/connector_hidden_states/independent.py \
  --raw /features/connector-window.pt --output /results/b-independent-ordinary-new \
  --transport ordinary

uv run --no-project .venv/bin/python examples/connector_hidden_states/independent.py \
  --raw /features/connector-window.pt --output /results/b-independent-mooncake-new \
  --transport mooncake
```

Set TARGET_MODEL/DRAFT_MODEL for both; set MOONCAKE_MASTER_ADDRESS for Mooncake.
The output directory must not exist. `--cpu-preflight` prepares features or checks
transport without launching training. Mooncake uses a cross-process exact-input
check, then the native Trainer directly reads TQ and clears consumed keys. The
initial feature file remains part of this two-stage workflow. Only use trusted
bundles: input is loaded with `weights_only=False`.

## Evidence boundaries

- The original unsynchronized cross-slot hidden-state discrepancy was not resolved.
  Later exact samples under synchronization do not close that issue.
- The live Co-Train evidence has eight exact consumer reads and successful master
  put/get/delete telemetry, but producer put/release INFO lines were missing from
  the main log. That logging gap remains recorded.
- A separate fixed-RNG/anchor experiment found identical inputs, anchors, full loss
  records and 63/63 model tensors initially and after each of two updates for
  ordinary versus Mooncake transport of the same B input. It does not validate
  upstream feature correctness or full Adam state.
- Uniform-warmup Window32 common-extraction API wall time was 1.130560s for B and
  1.200613s for A, one batch each. Both reduced output-related bytes by about half
  relative to Full64; worker allocator peaks did not fall correspondingly. These
  are not steady-state rankings, and the common A measurement was not its native
  real-drafter path. Co-Train step times also include different publication versus
  checkpoint work and cannot rank the architectures.

## Local validation

The branch contains portable CPU tests for Connector payload conversion, extractor
identity, TQ exact reads/corruption/release, shared explicit-window conversion,
agent-loop step propagation, native collection scheduling and replay. The shared
optional Runner fallback can also be contract-tested by setting
`SPECO_RUNNER_CAPTURE_SOURCE` to a compatible A source file; such CPU coverage is
not B GPU validation.

Runtime source is drawn from the verified B deployments with formatting, license
headers and one dynamic-config local type annotation. Server-specific defaults,
models, data, credentials and experiment logs are excluded. No new GPU run was
performed for publication of this branch.

Delivery checks on 2026-10-22: 178 related CPU tests passed without skips; all
repository pre-commit hooks passed. The portable Co-Train launcher passed Bash
syntax checking, and the Independent ordinary example passed CPU preflight on
the archived Connector input. These checks do not close the deferred B numerical
gate or validate the later vLLM e8f763c commit on GPU.
