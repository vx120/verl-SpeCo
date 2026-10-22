# Copyright 2026 Individual Contributor: vx120
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("backend,expected", [("connector", False), ("runner", True)])
def test_training_algorithm_does_not_make_extractor_a_serving_dspark(backend, expected):
    source = (
        Path(__file__).resolve().parents[2] / "verl_speco/integration/vllm_runtime.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_speco_is_dspark_algorithm"
    )
    namespace = {
        "_load_env_drafter_config": lambda: {
            "training": {"vllm_hidden_capture_backend": backend}
        },
        "_drafter_algorithm": lambda config: "DSPARK",
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    worker = SimpleNamespace(
        _speco_draft_method=lambda: "", _speco_resolve_draft_model=lambda: (None, None)
    )
    assert namespace["_speco_is_dspark_algorithm"](worker) is expected
