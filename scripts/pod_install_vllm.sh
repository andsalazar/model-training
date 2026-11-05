#!/bin/bash
# Install vLLM into the pod's local overlay disk.
#
# The first attempt installed into /workspace, which is a network volume, and pip
# spent 94 minutes thrashing on it. The overlay disk is local and much faster, so
# the venv lives there and only the model stays on the volume.
set -u
exec > /workspace/vllm_install.log 2>&1
set -x

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR=/root/.cache/pip

# Reuse the torch already installed on the volume so vLLM does not re-download it.
/workspace/venv/bin/pip install --no-deps vllm
/workspace/venv/bin/pip install vllm

echo "=== verify ==="
/workspace/venv/bin/python -c "import vllm; print('vllm', vllm.__version__)"
/workspace/venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "INSTALL_DONE"
