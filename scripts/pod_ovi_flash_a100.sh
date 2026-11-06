#!/usr/bin/env bash
# Build FlashAttention from source for A100 (sm_80) against torch 2.6.0+cu124.
#
# There is no prebuilt wheel for this torch/Python combination, so pip falls back to
# a source build and fails because build isolation hides torch from setup.py. The fix
# is --no-build-isolation plus nvcc on PATH. sm_80 needs only one arch, so the build
# is far cheaper than the Blackwell sm_120 attempt that kept getting OOM-killed.
set -euo pipefail

source /workspace/ovi-env/bin/activate

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0"
# One arch and 4 jobs: the Blackwell attempt at 16 jobs was OOM-killed repeatedly.
export MAX_JOBS=4

echo "=== nvcc ==="
which nvcc || { echo "nvcc not found"; ls /usr/local/ | grep -i cuda; }
nvcc --version | tail -2

echo "=== torch ==="
python -c "import torch; print(torch.__version__); import torch.utils.cpp_extension as e; print('CUDA_HOME', e.CUDA_HOME)"

echo "=== build ==="
cd /workspace
if [ ! -d flash-attention ]; then
  git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
fi
cd flash-attention
pip install --no-cache-dir -v . --no-build-isolation 2>&1 | tail -25

echo "=== verify ==="
python - <<'PY'
import torch
import flash_attn
print("flash_attn", flash_attn.__version__)
q = torch.randn(2, 64, 8, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn.flash_attn_func(q, q, q)
torch.cuda.synchronize()
print("flash_attn kernel OK", tuple(out.shape))
PY
echo "=== flash build complete ==="
