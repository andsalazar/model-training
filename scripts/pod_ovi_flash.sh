#!/usr/bin/env bash
# Install a FlashAttention build that works with torch 2.7.1+cu128 on sm_120.
#
# Ovi hard-asserts FLASH_ATTN_2_AVAILABLE in its attention module, so unlike the
# README's "optional" framing it is required. The earlier build failed because it was
# compiled against torch 2.8.0's C++ ABI while the venv ran 2.6.0. Now the venv runs
# 2.7.1+cu128, so a matching build is needed.
#
# Strategy: try a prebuilt wheel first (fast), fall back to compiling from source.
# Compiling needs ninja and takes 20-40 minutes on this hardware.
set -uo pipefail

source /workspace/ovi-env/bin/activate

echo "=== torch in venv ==="
python -c "import torch; print(torch.__version__)" 2>&1 | grep -v Warning

echo "=== ensure ninja and build tools ==="
pip install --no-cache-dir ninja packaging 2>&1 | tail -2
which ninja || echo "ninja not on PATH"
which nvcc || echo "nvcc not on PATH"
nvcc --version 2>/dev/null | tail -2 || true

echo "=== attempt prebuilt wheel ==="
# Dao-AILab publishes wheels for recent torch/CUDA combinations. If one matches
# 2.7.1+cu128 it installs in seconds; otherwise this fails and we compile.
pip install --no-cache-dir flash_attn 2>&1 | tail -6
if python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>/dev/null; then
  echo "PREBUILT_OK"
  python - <<'PY'
import torch
from flash_attn import flash_attn_func
q = torch.randn(2, 64, 8, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q)
torch.cuda.synchronize()
print("flash_attn kernel OK", out.shape)
PY
  exit 0
fi

echo "=== prebuilt failed, compiling from source ==="
cd /workspace
if [ ! -d flash-attention ]; then
  git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
fi
cd flash-attention
# MAX_JOBS controls parallelism; 112 vCPUs is plenty but memory is the real limit.
MAX_JOBS=16 pip install --no-cache-dir . 2>&1 | tail -15

echo "=== verify ==="
python - <<'PY'
import torch
from flash_attn import flash_attn_func
q = torch.randn(2, 64, 8, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q)
torch.cuda.synchronize()
print("flash_attn kernel OK", out.shape)
PY
