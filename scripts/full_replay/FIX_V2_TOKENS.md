# 修复 V2 重放时的无效草稿 token ID

根因有两处，关闭调度的配置本身不足以解决：

1. verl 的 `build_cli_args_from_config` 丢弃布尔 False，导致 `async_scheduling=false` 没有传成 vLLM 的否定参数，引擎可能自动启用异步调度。修复在两条服务器启动路径的 CLI 解析后、构建引擎前显式设置 `args.async_scheduling=False`。
2. 本地 vLLM V2 的 `DraftTokensHandler.set_draft_tokens` 只在 structured-output 请求中执行 CPU copy，普通文本请求会在 `get_draft_tokens` 返回 -1。修复仅在完整重放时，通过 handler 局部代理请求原有 CPU copy 路径；原 batch、采样及 grammar 状态不变。沿用原有 stream/event 同步，不用纠正 token 填补拒绝提案。

补丁针对 `c6d7693` 的完整重放实现制作，只含新模块与 runtime 安装代码，不包含本地已暂存的整组回退。

把本目录的 `fix_v2_proposal_tokens.patch` 复制到服务器。在服务器仓库根目录执行（补丁路径换成实际位置）：

```bash
cd /model/ljh/draaft-opd_VAT/verl-SpeCo
git apply --check /path/to/fix_v2_proposal_tokens.patch
git apply /path/to/fix_v2_proposal_tokens.patch
```

结束本次失败作业并重新创建该作业的 rollout/Ray actors。已有 worker 不会自动加载补丁；不要复用旧 feature buffer。原有真实 token 校验保留，不能删掉异常或将 -1 当作合法提案训练。

再运行本目录更新后的脚本，首次先用 1 个重放块、2 个 RL 步：

```bash
bash run_qwen3-8b-drafter_opd_dspark_full_replay_vllm.sh \
  trainer.total_training_steps=2 \
  actor_rollout_ref.rollout.drafter.training.drafter_opd.replay_max_blocks_per_sample=1
# DFlash 使用对应的 dflash 脚本，其他参数相同。
```

预期启动日志包含：
- `[speco opd replay] forcing engine async_scheduling=False after CLI parsing`
- `[speco opd replay] V2 real proposal CPU copy installed`

本地验证使用实际 V2 handler/verl CLI builder 的源码及 CPU 替身，覆盖普通文本和 structured-output、7/15 token 块、真实提案复制、关闭模式和幂等安装。不能替代服务器 GPU smoke。


## 第二次修复：CUDA 子进程入口与新请求 padding

第一次补丁还存在 CUDA 安装缺口：`patch_vllm_worker_proc_entrypoint` 在 Ascend 探测为 false 时直接返回，CUDA spawn worker 因而没有安装 token copy。现已取消这个通用入口对 Ascend 的依赖，并在 `SpecoVLLMColocateWorkerExtension.__new__` 再安装一次（幂等）。

此外，scheduler 为新加入/恢复的请求插入全 -1 的虚拟推测块。只有当 request 出现在 scheduled_new_reqs、全部提案均为 -1、且该步只生成一个 target token 时，trace 才记录 drafts=0，不把这个填充当作真实提案；普通请求无效 ID、混合有效/无效 ID 或已接受 token 的异常仍然报错。错误现在携带 request、generated、drafts、admitted 和 token 列表。

已应用第一次修复的服务器，复制 `apply_v2_replay_cuda_fix.py` 后执行 `python3 /path/to/apply_v2_replay_cuda_fix.py`，默认目标就是 `/model/ljh/draaft-opd_VAT/verl-SpeCo/verl_speco`。也可复制 `fix_v2_replay_cuda.sh` 后用 bash 执行。重复执行不会重复插入，变更前备份源码。

最新组合补丁 `fix_v2_proposal_tokens.patch` 已包含两次修复，适用于未打补丁的 c6d7693；已打旧补丁时请用补充 installer，避免重复 git apply。首次安装时 `apply_v2_replay_fix.py` 需要与 `apply_v2_replay_cuda_fix.py` 放在同一目录。

本地最新结果：41 项 trace/handler/CUDA-worker 入口定向测试通过，包含实际 WorkerProc hook 的非 Ascend 路径与新请求/恢复请求 padding；组合 installer 在 c6d7693 副本安装及重复运行检查通过。CUDA/Triton 与 Ray 多进程在测试中有替身，尚未实际完成服务器 GPU smoke。
