#!/bin/bash
# Bootstrap the temporary EU-RO-1 pod so explore15 can run.
#
# This pod has no network volume, so unlike the persistent pod it has to install
# vLLM and download the model from scratch. Everything lands in /workspace so it
# survives a container restart within the pod's lifetime.
set -u
exec > /workspace/setup.log 2>&1
set -x

export DEBIAN_FRONTEND=noninteractive
export PATH=/workspace/venv/bin:$PATH
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "=== python ==="
python3 -V

echo "=== venv ==="
python3 -m venv /workspace/venv
/workspace/venv/bin/python -m pip install --upgrade pip wheel setuptools

echo "=== torch ==="
/workspace/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "=== vllm ==="
/workspace/venv/bin/pip install vllm

echo "=== misc ==="
/workspace/venv/bin/pip install transformers safetensors numpy pytest httpx

echo "=== model ==="
/workspace/venv/bin/pip install huggingface_hub
/workspace/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3.5-0.8B", local_dir="/workspace/qwen")
print("MODEL_AT", p)
PY

echo "=== versions ==="
/workspace/venv/bin/python -c "import torch, vllm, transformers; print('torch', torch.__version__); print('vllm', vllm.__version__); print('transformers', transformers.__version__); print('cuda', torch.cuda.is_available())"

echo "SETUP_DONE"
