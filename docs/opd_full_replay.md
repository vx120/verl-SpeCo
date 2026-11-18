# Draft-OPD 完整提案分支重放

2026-11-17。实现范围为 GPU、vLLM、legacy trainer（`trainer.use_v1=false`）、FSDP/FSDP2 actor、old-logprob hidden 采集、DFlash 和标准 anchor-as-first DSpark。代码未修改参考仓库或 vLLM fork。

## 数据路径

1. scheduler 在更新 request 前复制有效 `draft_token_ids`，因此 target 纠正、停止截断或 request 删除不会覆盖原提案。无效 grammar 尾项不进入提案列表。
2. old-logprob collection 先按原策略选择源轨迹，再从其验证块中均匀抽取有限数量的完整块。每个分支的 prompt 为原轨迹到 anchor 的全部真实 token，response 为完整原草稿，包含首拒和后续 token；不使用 target 的纠正 token 替换拒绝提案。
3. 将这些分支追加到同一次 actor `compute_log_prob` 调用中。actor 对分支完整因果前缀重新前向，既有 forward-hook 采集 aux hidden 和 post-norm final hidden。这里不复用纠正轨迹在分歧后的状态，也不需要另加载 target 模型。
4. 原 PPO 行始终位于前部，重放行及为 DP 整除追加的行在返回 PPO 前被裁掉；logprob/entropy 的回答宽度恢复到原宽度。重放行只通过 drafter feature collection 入库，不进入奖励、优势、PPO 更新或 rollout AL。
5. 每个分支 feature 带 `kind=draft_branch_replay`，训练只选择它自己的锚点，并监督完整提案。DFlash 复用该 anchor 的并行起草；DSpark 的 Markov predecessor 来自原提案前缀。target 分布来自这条草稿分支的 final hidden，E2E-TV 对全部草稿位置计算。

因此首拒之后的位置也参与 loss 和梯度。该实现重放的是实际采样的提案分支，并不穷举所有可能提案，也不将单路径 TV 连乘宣称为严格的完整期望 AL。

## 配置和启动

YAML 总开关保持关闭以兼容旧样本；两个 OPD GPU 包装脚本显式开启完整模式。

```yaml
drafter_opd:
  enable: true
  capture_verify_trace: true
  replay_full_suffix: true
  replay_max_blocks_per_sample: 32  # 每条选中的源轨迹；0 表示全部合格块
  anchor_planner: trace
  first_reject_weight: 1.0
  loss: e2e_tv
```

完整模式要求 full_vocab、trace、E2E-TV 和 fixed_exp_decay 权重模式，不接受 CE、mixed 或 VAT 组合。CE 的 hard label 在分支上是草稿提案，不能直接拿它来模仿被拒 token。截断模式及既有 VAT 实现仍可通过关闭完整模式使用。

新增的 actor 前向会增加计算和显存，32 是可调的上限，不是实机调优结果。每个完整分支使用一个有效锚点，避免每行填充 256 个无效 draft 块。完整模式的 anchor 预算由 `replay_max_blocks_per_sample` 控制；原 `dflash_num_anchors` / `dspark_num_anchors` 用于截断模式。

完整模式要求 vLLM `engine_kwargs.vllm.async_scheduling=false`。本地 vLLM 的异步 scheduler 只保留 `-1` 占位符，真实提案留在 GPU worker；非异步路径才会将提案传回 scheduler。包装脚本显式关闭该选项，runtime 在未指定时补齐 false，遇到显式 true 则报错。消融启动器对所有 case 同样关闭异步 scheduling，避免吞吐比较混入调度差异；该开关可能影响实际 rollout 速度。

先使用新的 Ray actors 做小规模 smoke。路径和设备数按基础示例配置：

```bash
bash examples/run_qwen3-8b-drafter_opd_dflash_vllm.sh \
  trainer.n_gpus_per_node=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.drafter.training.drafter_opd.replay_max_blocks_per_sample=1 \
  trainer.total_training_steps=2 trainer.test_freq=-1

# 同一组种子比较截断和完整模式，先只打印启动命令。
bash examples/ablations/run_qwen3-8b_vat_opd.sh --dry-run dflash replay
# 完整模式的 32 / 8 个块预算比较：
bash examples/ablations/run_qwen3-8b_vat_opd.sh --dry-run dspark replay_budget
```

旧 rollout workers 不会提供提案 token；完整模式遇到这种轨迹会明确报错，不会退化为截断训练。需退出旧作业并重新创建 rollout/Ray actors。不要把旧 feature buffer 与新模式混用。

训练及 actor hidden 采集窗口均在示例中设为 2048；它们限制 drafter 的可见 context，但 actor 分支前向仍读取完整原前缀。窗口若切掉 anchor 或任何提案位置，训练会报错。EOS/长度导致实际提交截断的块整块排除；不把停止后位置伪装成已验证样本。多轮轨迹保留 tool token，并核对原前缀摘要和已接受 token。

## 验证边界

CPU 测试覆盖提案快照、无效尾项、分支与纠正轨迹分歧、完整 proposal mask、首拒后梯度、DFlash/DSpark dense loss 和梯度参照，以及真实 old-logprob hook 对重放行和回答宽度的裁剪。hook 测试使用真实 DataProto / padding 适配代码，并用等价 Torch 索引替代 flash-attn 的 CPU 不可用原语；actor RPC 和分布式 feature transport 使用替身。

尚未运行 GPU 多 rank、实际 vLLM/Ray worker、显存峰值和耗时测试，也没有新的 AL 结论。完整 trainer 合约测试在本机受可选依赖栈限制跳过；跳过不计为通过。

本次 CPU 验证：核心 loss / trace / replay / collection / old-logprob 测试 **184 passed**，另有 1 项 PP whole-ref 合约测试在导入 FSDP engine 时因缺少 `peft` 失败；OPD/VAT backend 与 vLLM runtime 定向测试 **59 passed / 228 deselected**；启动矩阵与 Hydra 组合 **13 passed**。合计 256 项通过，并不表示更广的合约套件全通过。三份重放启动脚本通过 Bash 语法检查；修改的 Python 文件通过 Ruff。

```bash
python -m pytest tests/unit/test_opd_replay.py tests/unit/test_drafter_opd.py \
  tests/unit/test_vat_loss.py tests/integration/test_verify_trace_runtime.py \
  tests/integration/test_collection_transport.py \
  tests/integration/test_oldlogprob_runtime_contract.py -q
python -m pytest tests/integration/test_dspark_trainer_backend.py \
  tests/integration/test_vllm_runtime_contract.py -k 'opd or vat' -q
python -m pytest tests/examples/test_vat_opd_ablations.py -q
```

测试环境需要 `VERL_SPECO_UPSTREAM_ROOT` 指向本地参考 verl；Windows 的 Bash 用 `SPECO_TEST_BASH` 指定。Windows 沙箱内的 asyncio transport 测试阻塞后，相关组在沙箱外重新运行得到上述结果，没有启动 Ray 集群或训练作业。
