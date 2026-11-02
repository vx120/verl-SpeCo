#!/usr/bin/env bash
set -euo pipefail

export SPECO_PYTHON=${SPECO_PYTHON:-/data/logprob-validation/.venv/bin/python}
export SPECO_VERL_SOURCE=${SPECO_VERL_SOURCE:-/data/segments-validation/verl-legacy-6eeb5711}
export SPECO_VERL_COMMIT=6eeb5711babe178ca3cc3caba7c9f158efa9fcd6
export SPECO_REPO_ROOT
SPECO_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if ! test -f "$SPECO_VERL_SOURCE/verl/workers/fsdp_workers.py"; then
  archive=$(mktemp --suffix=.tar.gz)
  trap 'rm -f "$archive"' EXIT
  curl -fL --connect-timeout 15 --max-time 75 \
    "https://codeload.github.com/verl-project/verl/tar.gz/$SPECO_VERL_COMMIT" -o "$archive"
  mkdir -p "$SPECO_VERL_SOURCE"
  tar -xzf "$archive" --strip-components=1 -C "$SPECO_VERL_SOURCE"
fi

"$SPECO_PYTHON" - <<'PY'
import json
import os
import sysconfig
from pathlib import Path

source = Path(os.environ['SPECO_VERL_SOURCE']).resolve()
path = Path(sysconfig.get_path('purelib')) / 'verl_compat.pth'
path.write_text(f'import sys; sys.path.insert(0, {str(source)!r})\n')
(source / 'speco_source.json').write_text(json.dumps({
    'repository': 'https://github.com/verl-project/verl',
    'commit': os.environ['SPECO_VERL_COMMIT'],
}, indent=2))
PY

"$SPECO_PYTHON" - <<'PY'
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ['SPECO_REPO_ROOT'])
import verl
import verl.workers.fsdp_workers
import verl_speco.trainer.speco_ray_trainer
from verl.trainer.main_ppo import TaskRunner

assert hasattr(TaskRunner, 'add_reward_model_resource_pool')
versions = {name: metadata.version(name) for name in ('transformers', 'torch', 'vllm', 'ray')}
versions.update(verl_version=verl.__version__, verl_source=verl.__file__,
                verl_commit=os.environ['SPECO_VERL_COMMIT'])
Path(os.environ['SPECO_VERL_SOURCE'], 'speco_environment.json').write_text(json.dumps(versions, indent=2))
print('Training imports and scheduler API OK')
print(json.dumps(versions, indent=2))
PY
