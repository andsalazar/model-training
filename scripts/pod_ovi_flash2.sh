#!/usr/bin/env bash
# Build FlashAttention from source against torch 2.7.1+cu128 for sm_120.
#
# Two problems in the first attempt:
#   1. nvcc was not on PATH. The toolkit is at /usr/local/cuda-12.8 but the venv's
#      PATH does not include it.
#   2. pip's build isolation created a fresh env with no torch, so setup.py failed
#      with "No module named 'torch'". --no-build-isolation makes the build use the
#      venv's own torch, which is what a CUDA extension needs.
#
# Ovi hard-asserts FLASH_ATTN_2_AVAILABLE, so this is required, not optional.
set -uo pipefail

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=16

source /workspace/ovi-env/bin/activate

echo "=== toolchain ==="
nvcc --version | tail -2
python -c "import torch; print('torch', torch.__version__)" 2>&1 | grep -v Warning

echo "=== build flash-attention ==="
cd /workspace/flash-attention
# The repo was cloned in the previous attempt; reuse it.
git log --oneline -1 2>/dev/null || true
pip install --no-cache-dir --no-build-isolation . 2>&1 | tail -20

echo "=== verify ==="
python - <<'PY'
import torch
import flash_attn
print("flash_attn", flash_attn.__version__)
from flash_attn import flash_attn_func
q = torch.randn(2, 64, 8, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q)
torch.cuda.synchronize()
print("flash_attn kernel OK", tuple(out.shape))
PY
