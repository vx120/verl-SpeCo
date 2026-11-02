# Capture training environment

The tested capture path uses the legacy verl FSDP workers. The upstream breaking
change `044bbba27e5861e1212727aba67813cccb2db291` removed those workers. Current
release/v0.8.0 and v0.8.0 tag sources do not provide the required old import path;
pinning only the version string does not recover the previous environment.

Run `bash scripts/setup_capture_training_env.sh`. It selects the official parent
commit `6eeb5711babe178ca3cc3caba7c9f158efa9fcd6`, which contains both the legacy
workers and `TaskRunner.add_reward_model_resource_pool`. The source reports
`0.8.0.dev`. No upstream source files or system packages are patched. The earlier
0.7.0 trial is not selected because its scheduler API is too old.

Defaults:

- Python: `/data/logprob-validation/.venv/bin/python` (`SPECO_PYTHON`)
- Source: `/data/segments-validation/verl-legacy-6eeb5711` (`SPECO_VERL_SOURCE`)

A `verl_compat.pth` file in the task venv selects this persistent source for the
launcher and subprocesses. The script verifies imports and the scheduler API,
and records source identity and effective dependencies inside the source folder.
The current system stack is transformers 5.19.0, torch 2.13.0+cu130, vLLM 0.32.0,
and Ray 2.59.0. These packages are left intact. Revalidate runtime compatibility
after changing the container image.

Use `PYTHON_BIN=/data/logprob-validation/.venv/bin/python` with
`examples/run_windowed_hidden_capture_smoke.sh` for the limited runtime check.
It runs three global steps, with drafter collection/training/publication at
interval two and rollout concurrency four. Keep an external timeout and clean
up the run's Ray processes afterward. This checks the Qwen3 text training path;
it does not validate multimodal training or full-vocabulary OPD scheduling.

To stop selecting this source, remove `verl_compat.pth` from the task venv's
site-packages. The system installation, model files and datasets remain intact.

The SpeCo compatibility mixin forwards the `mode` argument only to workers that
accept it. The pinned legacy worker supports automatic synchronization only; an
explicit alternative mode raises an error instead of silently changing behavior.
