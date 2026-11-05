#!/bin/bash
# Install vLLM on the pod's local NVMe disk.
#
# /workspace is a network volume and throttles pip to ~1.4 MB/s, which made a
# vLLM install take over 90 minutes. The host has a 1.2 TB local NVMe mount, so
# the venv goes there and only the model stays on the volume.
set -u
exec > /workspace/local_install.log 2>&1
set -x

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR=/root/.cache/pip
export PATH=/local/venv/bin:$PATH

mkdir -p /local
python3 -m venv /local/venv
/local/venv/bin/python -m pip install --upgrade pip wheel setuptools

# torch first, from the CUDA 12.8 index, so vLLM does not pull a different build.
/local/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128

/local/venv/bin/pip install vllm

/local/venv/bin/pip install transformers safetensors numpy pytest httpx

echo "=== verify ==="
/local/venv/bin/python -c "import vllm; print('vllm', vllm.__version__)"
/local/venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "LOCAL_INSTALL_DONE"
