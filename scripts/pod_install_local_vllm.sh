#!/bin/bash
# Install vLLM on the pod's local overlay disk, bypassing the network volume.
#
# The network volume (/workspace) throttles reads to ~1 MB/s, which made both a
# venv copy and a pip install stall. The local overlay is 80 GB and fast.
set -u
LOG=/workspace/install_local.log
: > "$LOG"

{
  echo "=== START $(date -u +%H:%M:%S) ==="
  export PIP_CACHE_DIR=/local/pipcache
  mkdir -p /local/pipcache

  python3 -m venv /local/venv
  /local/venv/bin/pip install --upgrade pip -q 2>&1 | tail -2

  echo "=== INSTALL VLLM ==="
  time /local/venv/bin/pip install vllm 2>&1 | tail -25
  echo "pip_rc=$?"

  echo "=== VERIFY ==="
  timeout 180 /local/venv/bin/python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -4
  timeout 180 /local/venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -4

  echo "INSTALL_LOCAL_DONE"
} >> "$LOG" 2>&1
