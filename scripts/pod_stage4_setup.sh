#!/bin/bash
# Stage 4 setup: vLLM for eval, plus peft/accelerate for training.
#
# Everything goes on the pod's local overlay disk. /workspace is a network volume
# that throttled a vLLM install to ~1 MB/s in Stage 2 (94 minutes versus 3.5 on
# local disk), and the same applies to any large install.
set -u
LOG=/workspace/stage4_setup.log
: > "$LOG"

{
  echo "=== START $(date -u +%H:%M:%S) ==="
  export PIP_CACHE_DIR=/root/.cache/pip
  mkdir -p /local

  if [ -x /local/venv/bin/python ]; then
    echo "venv already present at /local/venv"
  else
    python3 -m venv /local/venv
    /local/venv/bin/pip install -q --upgrade pip
    echo "=== VLLM (eval side) ==="
    time /local/venv/bin/pip install -q vllm 2>&1 | tail -5
  fi

  echo "=== TRAINING DEPS ==="
  time /local/venv/bin/pip install -q peft accelerate datasets 2>&1 | tail -8

  echo "=== VERIFY ==="
  timeout 180 /local/venv/bin/python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -3
  timeout 180 /local/venv/bin/python -c "import peft; print('peft', peft.__version__)" 2>&1 | tail -3
  timeout 180 /local/venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -3

  echo "STAGE4_SETUP_DONE"
} >> "$LOG" 2>&1
