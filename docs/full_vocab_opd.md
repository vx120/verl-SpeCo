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
