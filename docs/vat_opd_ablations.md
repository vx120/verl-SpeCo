# VAT / Draft-OPD 消融

2026-11-17：默认配置清理及实验启动器。没有运行 GPU 训练，没有新的 AL 收益结论。

本次仅修改 `verl-SpeCo`。OPD 的 Python fallback 和 YAML 默认值统一为 `anchor_planner=trace`、`first_reject_weight=1`；`mixed` 及倍权选项保留。`random_anchor_ratio=0.25` 只在 mixed 模式生效。DFlash VAT 包装脚本去掉 front-position 加权，普通基础示例和 D-PACE 的位置权重保持各自原有配置。

两个 OPD 包装脚本使用 greedy 起草、TV 温度 1、完整提案重放和 2048-token 窗口，actor hidden 采集窗口也设为 2048。完整模式每条源轨迹最多采样 32 个完整块，每个 branch feature 使用一个锚点；原 256 锚点预算用于截断模式。这些预算并非已证明最优，可以通过末尾 Hydra 参数覆盖。DSpark 的训练窗口可以用 0 表示不限长，DFlash 当前不能用 0 表示不限长。

2026-11-17 已接入 [完整提案分支重放](opd_full_replay.md)：记录原提案，actor 在分支前缀上重新计算 target hidden，额外行不进入 PPO。旧截断模式保留为 `opd_truncated`。概率起草时，单步分布重叠对应平均接受概率，但不能仅凭采样方式和温度一致，就把单路径 TV 连乘解释成完整 rollout 的精确期望 AL。VAT 的 argmax 接受代理和 DSpark 条件标签保持不变。

## 启动

沿用基础示例的 `MODEL_PATH`、`DRAFTER_PATH`、`TRAIN_FILE`、`TEST_FILE` 环境变量；两种模型必须分别使用匹配的 drafter checkpoint。启动器可以从任意目录调用，内部切回仓库根目录。默认每组使用种子 42、43、44，按种子、case 顺序串行运行，失败即停止。

```bash
# 先检查命令；不会创建目录、导入训练框架或启动训练。
bash examples/ablations/run_qwen3-8b_vat_opd.sh --dry-run dflash core \
  trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.tensor_model_parallel_size=2

# 正式运行，路径不要重复使用；每个 case/seed 使用独立目录。
export SPECO_ABLATION_OUTPUT=/path/to/results/dflash_core_20261117
export SPECO_ABLATION_SEEDS="42 43 44"
bash examples/ablations/run_qwen3-8b_vat_opd.sh dflash core \
  trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.tensor_model_parallel_size=2

# 换成匹配的 DSpark checkpoint 后单独运行。
export SPECO_ABLATION_OUTPUT=/path/to/results/dspark_sampling_20261117
bash examples/ablations/run_qwen3-8b_vat_opd.sh dspark sampling \
  trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.tensor_model_parallel_size=2

# 也可以只运行一个 case。
bash examples/ablations/run_qwen3-8b_vat_opd.sh --dry-run dflash opd_reject2
```

基础示例的 GPU 数为 16、TP 为 2；按机器资源显式覆盖，不会自动探测 GPU。默认 core 为 9 个作业；`--list` 查看 case，`--help` 查看用法。`all` 包含全部 case，成本较高，不是默认组。

每个目录保存 `command.sh`、`revision.txt`、`train.log` 和 `checkpoints/`。目录已存在时报错，避免覆盖实验。`revision.txt` 包括 HEAD 与工作区状态；修改源码后应自行保存相应版本。停止后的作业不会自动续跑，可用单 case 和 `SPECO_ABLATION_SEEDS` 在新输出目录重新启动。

## 共同控制条件

全部 case 使用 full_vocab、target T=0.6 / top-p=1 / top-k=-1、同步训练及发布、禁用 packing/data buffer、只保留最近一步样本、采集和训练间隔 2、每次训练 10 步、200 个 RL 步、最大回答长度 2048。每 100 步保存 checkpoint，不运行周期评测。其余配置继承基础示例。DFlash 使用 16-slot 块（15 个 draft），DSpark 使用 7 个 draft；比较在各自模型内完成，不直接将两种架构的 AL 排名。

为让 scheduler 获取完整提案，全部 case 统一设置 vLLM `async_scheduling=false`。这里的调度选项与 training/publish 同步选项不同，也会影响吞吐。完整模式不支持启用该异步调度；原独立 VAT 示例保持原有调度行为。

baseline 关闭 OPD/VAT，保留各模型原有主损失：DFlash CE，DSpark CE+L1。它与处理组保持相同的 full_vocab、窗口和锚点预算；不是历史 restricted_ce 启动配置。VAT 单独运行，DSpark VAT 启用 confidence head 并关闭原 confidence loss。其他 DSpark case 显式关闭 confidence 辅助损失。

末尾 Hydra 参数可以覆盖共同条件和 case 参数；覆盖时需记录偏离。种子和运行目录/名称由启动器在最后指定，不接受末尾参数覆盖；通过环境变量选择种子和输出根目录。种子设置涵盖数据顺序、actor minibatch、actor FSDP 与 vLLM engine seed（`rollout.engine_kwargs.vllm.seed`）；同一个作业的 rollout replicas 使用同一个 engine seed。不承诺 Ray、CUDA 或 drafter worker 的所有 RNG 都确定，也不保证不同采样方式消费相同随机数。

## 实验组

除表中改变外，以 `opd` 的完整重放 / trace / 权重 1 / 每条源轨迹最多 32 块 / 2048 窗口 / T=1 / greedy 为参照。

| 组 | case | 要回答的问题 |
| --- | --- | --- |
| `core` | baseline、vat、opd | 清理后的 VAT、完整重放 OPD 是否有独立收益 |
| `replay` | opd_truncated、opd | 截断与完整提案重放；同时记录额外 actor 开销 |
| `replay_budget` | opd、opd_replay8 | 完整重放每条源轨迹最多 32 与 8 个块 |
| `front`（仅 DFlash） | vat、vat_front | VAT 前 3 位权重 1 与 2 的差异 |
| `planner` | opd_truncated、opd_mixed | 截断模式下，纯 trace 与 25% 随机配额 |
| `reject` | opd、opd_reject2 | 首拒权重 1 与 2 |
| `anchors` | opd_truncated、opd_anchors64 | 截断模式下，窗口不变，256 与 64 锚点 |
| `window` | opd、opd_window512 | 锚点预算不变，2048 与 512 窗口 |
| `temperature` | opd、opd_t06、opd_prob_t1、opd_prob_t06 | OPD 的 TV 温度 1/0.6 × greedy/probabilistic |
| `sampling` | baseline、baseline_prob、opd_t06、opd_prob_t06 | 固定 OPD TV 温度 0.6，分离采样切换收益与 OPD 收益 |
| `legacy` | baseline_legacy、opd_legacy、baseline、opd | 旧配置组合与新配置组合，各配一个相同预算的 baseline |

`opd_legacy` 使用截断模式、mixed、首拒权重 2、64 锚点、512 窗口、TV 温度 0.6、greedy；`baseline_legacy` 仅将 baseline 的预算改成 64/512。512 窗口 case 同时将 hidden 采集窗口设为 512。legacy 比较有多个因素同时变化，只用于整体回归，不能归因到单个参数，也不复刻历史运行的全部服务器配置。完整模式不支持 mixed；planner 和 anchors 组因此保留在截断模式内比较。

## 评估顺序

先跑 core；有收益迹象后再跑单因素组。采样问题同时看 sampling 和 temperature，不能只将 greedy baseline 与 probabilistic OPD 比较。暂不加入 VAT+OPD 组合。

记录训练期间 AL 曲线、最后固定步数区间均值、触发训练时的耗时、整步耗时和峰值显存；区分未训练步与训练步。对同一组种子报告逐种子差值及均值/离散程度，不把 200 个相关 RL 步当作 200 次独立重复。

另将保存的 drafter checkpoint 放在相同冻结 target、相同评测 prompts 和采样配置下评估 AL、吞吐/延迟。在线 target 会随 RL 更新，仅凭训练日志均值不能隔离 drafter 的作用。本启动器负责训练矩阵；固定 target 评测需使用部署环境对应的评测入口，没有伪造自动评测结果。

## 本次验证

CPU 环境已运行以下两组测试，分别为 **68 passed / 1 deselected** 和 **35 passed / 81 deselected**：

```bash
# VERL_SPECO_UPSTREAM_ROOT 指向可导入的本地 verl 参考仓库。
# Windows 可用 SPECO_TEST_BASH 指定 Git Bash 的路径。
python -m pytest tests/examples/test_vat_opd_ablations.py \
  tests/unit/test_drafter_opd.py tests/unit/test_vat_loss.py \
  tests/config/test_speco_config_overlay.py -k 'not veomni' -q -p no:cacheprovider
python -m pytest tests/integration/test_dspark_trainer_backend.py \
  -k 'opd or vat' -q -p no:cacheprovider
```

第一组覆盖最终启动参数、含空格的输出路径、单因素矩阵、种子配对、包装脚本默认值，以及 baseline/VAT/OPD 的真实 Hydra 配置组合。Bash 测试将 `python3` 替换为只输出参数的函数，没有启动训练。第二组验证实际 backend 的 OPD/VAT 前向反向。六份相关 shell 脚本通过 `bash -n`，修改的 Python 文件通过 Ruff，`git diff --check` 通过。

初始完整配置测试中，VeOmni 用例因本地参考上游缺少 `actor.veomni.router_replay` 而失败，最终定向运行明确排除了该项，没有修补无关的 VeOmni 配置。Windows Hydra 的临时配置 searchpath 有警告，所列配置组合仍通过。GPU 显存、耗时、AL、多 rank 训练及固定 target 评测尚未验证。
