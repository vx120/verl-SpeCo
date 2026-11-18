"""Check final launch arguments across the matrix and nested Bash wrappers.

The python3 function below only emits argv; no training module is imported.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "examples/ablations/run_qwen3-8b_vat_opd.sh"
TRAINING = "actor_rollout_ref.rollout.drafter.training"


@pytest.fixture(scope="module")
def bash():
    executable = os.environ.get("SPECO_TEST_BASH") or shutil.which("bash")
    if executable is None:
        pytest.skip("Bash is unavailable")
    probe = subprocess.run([executable, "--version"], capture_output=True)
    if probe.returncode:
        pytest.skip("Bash cannot initialize in this environment")
    return executable


def _dry_commands(bash, model, group, output, seeds="42 43", *overrides):
    env = dict(
        os.environ, SPECO_ABLATION_SEEDS=seeds, SPECO_ABLATION_OUTPUT=output.as_posix()
    )
    result = subprocess.run(
        [bash, RUNNER.as_posix(), "--dry-run", model, group, *overrides],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert not output.exists()
    return [shlex.split(line) for line in result.stdout.splitlines()]


def _launch_overrides(bash, command):
    # Feed Bash source on stdin: Windows argv quoting otherwise rewrites literal
    # quotes in Hydra list values while crossing Python -> MSYS Bash.
    source = 'python3() { printf "%s\\0" "$@"; }; export -f python3;\n' + shlex.join(
        command
    )
    result = subprocess.run(
        [bash],
        input=source,
        text=True,
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    args = result.stdout.rstrip("\0").split("\0")
    assert args[:2] == ["-m", "verl_speco.main"]
    return args[2:]


def _final_config(bash, command):
    config = OmegaConf.load(ROOT / "verl_speco/config/speco_base.yaml")
    for override in OverridesParser.create().parse_overrides(
        _launch_overrides(bash, command)
    ):
        OmegaConf.update(config, override.key_or_group, override.value(), merge=False)
    return config


@pytest.mark.parametrize("model", ["dflash", "dspark"])
def test_matrix_survives_nested_wrappers_and_isolates_factors(bash, tmp_path, model):
    commands = _dry_commands(bash, model, "all", tmp_path / "results with spaces", "17")
    configs = {}
    for command in commands:
        config = _final_config(bash, command)
        name = config.trainer.experiment_name
        case = name.removeprefix(f"{model}_").removesuffix("_seed17")
        configs[case] = config
        training = OmegaConf.select(config, TRAINING)
        rollout = config.actor_rollout_ref.rollout
        assert config.data.seed == rollout.engine_kwargs.vllm.seed == 17
        assert config.actor_rollout_ref.actor.fsdp_config.seed == 17
        assert config.actor_rollout_ref.actor.data_loader_seed == 17
        assert name in config.trainer.default_local_dir
        assert "results with spaces" in config.trainer.default_local_dir
        assert rollout.temperature == 0.6 and rollout.top_p == 1 and rollout.top_k == -1
        assert rollout.engine_kwargs.vllm.async_scheduling is False
        assert training.publish_async is False and training.packing.enable is False
        assert training[f"{model}_loss_mode"] == "full_vocab"
        assert training.drafter_opd.enable == case.startswith("opd")
        assert training.drafter_opd.capture_verify_trace == case.startswith("opd")
        assert training.drafter_opd.replay_full_suffix == (
            case.startswith("opd")
            and case
            not in {"opd_truncated", "opd_anchors64", "opd_mixed", "opd_legacy"}
        )
        assert training.drafter_opd.replay_max_blocks_per_sample == (
            8 if case == "opd_replay8" else 32
        )
        assert training.drafter_opd.anchor_planner == (
            "mixed" if case in {"opd_mixed", "opd_legacy"} else "trace"
        )
        assert training.drafter_opd.first_reject_weight == (
            2 if case in {"opd_reject2", "opd_legacy"} else 1
        )
        assert training[f"{model}_num_anchors"] == (
            64 if case in {"opd_anchors64", "opd_legacy", "baseline_legacy"} else 256
        )
        assert training[f"{model}_max_window"] == (
            512 if case in {"opd_window512", "opd_legacy", "baseline_legacy"} else 2048
        )
        assert training.lk_temperature == (
            0.6 if case in {"opd_t06", "opd_prob_t06", "opd_legacy"} else 1
        )
        assert rollout.drafter.vllm.draft_sample_method == (
            "probabilistic"
            if case in {"baseline_prob", "opd_prob_t1", "opd_prob_t06"}
            else "greedy"
        )
        assert training[f"{model}_per_position_loss_weight"] == (
            "vat" if case.startswith("vat") else "fixed_exp_decay"
        )
        if model == "dspark":
            assert training.dspark_confidence_head_alpha == int(case == "vat")
            assert training.dspark_confidence_loss_alpha == 0
        else:
            assert training.dflash_front_position_weight == (
                2 if case == "vat_front" else 1
            )
            assert training.dflash_front_position_count == (
                3 if case == "vat_front" else 0
            )
    assert len(configs) == (16 if model == "dflash" else 15)


@pytest.mark.parametrize("model", ["dflash", "dspark"])
def test_sampling_group_pairs_seeds_and_forwards_overrides(bash, tmp_path, model):
    commands = _dry_commands(
        bash,
        model,
        "sampling",
        tmp_path / "dry-run",
        "42 43 44",
        f"{TRAINING}.{model}_num_anchors=128",
        "trainer.n_gpus_per_node=2",
    )
    assert len(commands) == 12
    configs = [_final_config(bash, command) for command in commands]
    assert len({config.trainer.default_local_dir for config in configs}) == 12
    for seed in (42, 43, 44):
        paired = [c for c in configs if c.data.seed == seed]
        assert len(paired) == 4
        assert {
            c.actor_rollout_ref.rollout.drafter.vllm.draft_sample_method for c in paired
        } == {"greedy", "probabilistic"}
        assert (
            sum(OmegaConf.select(c, f"{TRAINING}.drafter_opd.enable") for c in paired)
            == 2
        )
    assert all(c.trainer.n_gpus_per_node == 2 for c in configs)
    assert all(
        OmegaConf.select(c, f"{TRAINING}.{model}_num_anchors") == 128 for c in configs
    )


@pytest.mark.parametrize("model", ["dflash", "dspark"])
def test_launch_composes_against_available_upstream(bash, tmp_path, model):
    upstream = os.environ.get("VERL_SPECO_UPSTREAM_ROOT")
    if not upstream:
        pytest.skip("Set VERL_SPECO_UPSTREAM_ROOT for full Hydra composition")
    config_dir = tmp_path / "config"
    shutil.copytree(Path(upstream) / "verl/trainer/config", config_dir)
    for name in ("speco_base.yaml", "speco_trainer.yaml"):
        source = (ROOT / "verl_speco/config" / name).read_text(encoding="utf-8")
        source = source.replace("pkg://verl.trainer.config", config_dir.as_uri())
        (config_dir / name).write_text(source, encoding="utf-8")
    commands = _dry_commands(bash, model, "core", tmp_path / "outputs", "42")
    for command in commands:
        overrides = _launch_overrides(bash, command)
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            config = compose(config_name="speco_trainer", overrides=overrides)
        assert config.actor_rollout_ref.rollout.engine_kwargs.vllm.seed == 42
        assert config.actor_rollout_ref.actor.fsdp_config.seed == 42


@pytest.mark.parametrize("model", ["dflash", "dspark"])
@pytest.mark.parametrize("method", ["vat", "drafter_opd"])
def test_standalone_wrapper_defaults(bash, model, method):
    script = f"examples/run_qwen3-8b-{method}_drafter_{model}_vllm.sh"
    if method == "drafter_opd":
        script = f"examples/run_qwen3-8b-drafter_opd_{model}_vllm.sh"
    config = _final_config(bash, [bash, script])
    training = OmegaConf.select(config, TRAINING)
    assert training[f"{model}_loss_mode"] == "full_vocab"
    if model == "dflash":
        assert training.dflash_front_position_weight == 1
        assert training.dflash_front_position_count == 0
    if method == "drafter_opd":
        assert training.drafter_opd.anchor_planner == "trace"
        assert training.drafter_opd.first_reject_weight == 1
        assert training[f"{model}_num_anchors"] == 256
        assert training[f"{model}_max_window"] == 2048
        assert training.lk_temperature == 1
        assert (
            config.actor_rollout_ref.rollout.drafter.vllm.draft_sample_method
            == "greedy"
        )
    else:
        assert training[f"{model}_per_position_loss_weight"] == "vat"


@pytest.mark.parametrize(
    "args", [("bad", "core"), ("dflash", "bad"), ("dspark", "front")]
)
def test_invalid_matrix_request_fails_before_training(bash, args):
    result = subprocess.run(
        [bash, RUNNER.as_posix(), *args], cwd=ROOT, capture_output=True
    )
    assert result.returncode == 2
