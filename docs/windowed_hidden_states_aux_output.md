# vLLM AuxOutput window capture

Last updated: 11/02/2026

The vLLM integration uses the AuxOutput Connector on CUDA Model Runner V2.
SpeCo keeps the existing window planner, training sample layout and hidden transport.
Only the engine request/result adapter changes: the request window is placed in
`SamplingParams.extra_args.hidden_state_capture`, and the result is consumed from
`RequestOutput.aux_outputs.hidden_state_capture`.

Enable this source and disable the other collection sources:

```yaml
actor_rollout_ref:
  rollout:
    engine_kwargs:
      vllm:
        async_scheduling: false
    drafter:
      training:
        collect_hidden_states_from_vllm: true
        collect_hidden_states_from_sgl: false
        collect_hidden_states_from_old_logprob: false
        vllm_hidden_state_window:
          mode: fixed
          coordinate: response
          start: 1000
          end: 2000
          min_rows: 1000
```

SpeCo automatically enables `aux_output_config.enable_hidden_states` in the engine
configuration. When `training.use_logits` is enabled, it also defaults the engine's
`max_logprobs` to `training.logits_topk` (128 when omitted), so capture requests
fit the engine admission limit. An explicit `engine_kwargs.vllm.max_logprobs`
is preserved and must permit the requested top-k (`-1` means unlimited).

Set `VLLM_USE_V2_MODEL_RUNNER=1` before starting the job. When passing
fixed `start` and `end` fields through Hydra, use `+` if they are not present in the
base YAML, for example:

```bash
actor_rollout_ref.rollout.drafter.training.vllm_hidden_state_window.mode=fixed \
+actor_rollout_ref.rollout.drafter.training.vllm_hidden_state_window.start=1000 \
+actor_rollout_ref.rollout.drafter.training.vllm_hidden_state_window.end=2000
```

The window is zero-based and half-open. `[1000, 2000)` requests 1000 rows; absolute
hidden position is `prompt_len - 1 + response_index`. Ensure the generation budget
can reach the window. Early termination is reported through capture skip metrics.
`use_logits=true` keeps the existing window-aligned raw top-k logprob supervision.

The adapter copies sampling parameters before inserting request-local metadata,
so concurrent requests do not modify a shared sampling object. The engine owns
speculative rejection filtering and capture completion. SpeCo continues to verify
hidden positions, next-token labels, selected layers and final normalization when
building training samples. TQ/Mooncake transport is unchanged.

The previous Runner integration and its historical deployment measurements are
documented in `windowed_hidden_states_runner.md`. They are not performance results
for the AuxOutput implementation. Measure the deployed version using the vLLM
window benchmark before comparing throughput.

## Short training check

For the local Qwen3/EAGLE3 setup, run `bash examples/run_windowed_hidden_capture_smoke.sh`.
It limits the run to three global steps, collects/trains/publishes every two steps,
disables evaluation and checkpoint saving, and sets `rollout.max_num_seqs=4`.
The concurrency cap matters: capture admission reserves bytes for the configured
maximum sequence count, not just the current batch. A default of 1024 can exceed
the 256 MiB capture budget even when the actual rollout batch has four requests.
