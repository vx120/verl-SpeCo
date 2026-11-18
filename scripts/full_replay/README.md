# Draft-OPD 完整分支重放启动脚本

来源：
- DSpark：`D:/Downloads/20261116_010305/run_qwen3-8b-drafter_opd_dspark_vllm.sh`
- DFlash：`D:/workspace/draft-opd_VAT/scripts/20261116_010309/run_qwen3-8b-drafter_opd_dflash_vllm.sh`

原文件未改动；本目录只复制启用 OPD 的脚本，没有复制 baseline 和历史日志。

共同修改：开启 `replay_full_suffix=true`；每条选中源轨迹最多重放 32 个完整块；使用必需的 trace-only + E2E-TV；首拒权重显式设为 1；关闭 vLLM async_scheduling；显式使用 legacy trainer 和 greedy 起草；`trainer.save_freq=-1`（关闭周期 checkpoint 保存）。检查点和 rollout 输出目录、实验名称带 full_replay，且修正原 rollout 路径缺少分隔符的问题。

其余主要实验参数沿用源脚本：target 采样温度和 TV 温度均为 0.6，2 GPU / TP=2，200 RL 步，每 2 步进行 10 次 drafter 更新，训练窗口 512。未把 TV 温度改为 1 或改成概率起草。完整模式中每个分支只有一个有效锚点，块预算由 replay_max_blocks_per_sample 控制；原 num_anchors=64 不代表每条分支训练 64 个块。

脚本沿用原 Linux 模型/数据/仓库路径和 GPU 编号；部署时按实际机器确认。复制到服务器后，用 bash 执行对应脚本。首次可在命令末尾追加：

```bash
trainer.total_training_steps=2 actor_rollout_ref.rollout.drafter.training.drafter_opd.replay_max_blocks_per_sample=1
```

**运行代码必须包含 c6d7693 的完整重放实现或兼容版本。** 当前本地工作区暂存了对该实现的回退，本次只生成脚本，没有取消回退。启动检查会拒绝缺少 `verl_speco/integration/opd_replay.py` 的代码目录。不能在当前回退后的实现上仅添加开关就认为完整重放生效。请用新的 rollout/Ray actors 启动作业，不复用旧 feature buffer。

用户在脚本后传入的 Hydra 参数具有最终覆盖权；如果追加 save_freq 等参数，会覆盖本目录默认值。

## V2 token 回传修复

运行前还须应用本目录的 `fix_v2_proposal_tokens.patch`，具体见 `FIX_V2_TOKENS.md`。这同时处理 verl 丢弃 False 参数和 V2 普通请求不回传真实提案的问题。

## 纯文本 metadata 误判修复

若在 actor 批次构造阶段遇到 `Draft-OPD branch replay currently supports text-only position IDs`，旧版 guard 同时检查 metadata 字段存在性和 position_ids 维度，错误无法区分二者。纯文本 agent loop 也会输出全为空字典的 multi_modal_inputs 列。

运行 `python3 /path/to/apply_replay_text_metadata_fix.py`（默认服务器代码目录），允许 None/空字典 metadata；保留真实非空多模态输入限制，并单独打印异常 position_ids 的 shape。本地已在二维纯文本 position_ids 上复现旧错误，同一数据修复后通过，相关定向回归共 55 passed。仍需服务器 GPU smoke。
