#!/bin/bash
# Build FlashAttention 2 for sm_80 only.
#
# Two earlier attempts failed for a reason that was not visible: the setup script piped
# the build through `tail -5` / `tail -3`, so the compiler error never survived.
#
# Root cause: flash-attention's setup.py honours FLASH_ATTN_CUDA_ARCHS, not
# TORCH_CUDA_ARCH_LIST. Its default is "80;90;100;110;120", so setting
# TORCH_CUDA_ARCH_LIST=8.0 had no effect and the build compiled the whole kernel set
# five times over. On an A100 only sm_80 is needed.
set -uo pipefail

source /local/ovi-env/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
export FLASH_ATTN_CUDA_ARCHS="80"
export NVCC_THREADS=4
export MAX_JOBS=8

cd /local/flash-attention || exit 1

echo "--- preconditions ---"
which nvcc gcc ninja
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS"

echo "--- clearing stale objects ---"
rm -rf build dist

echo "--- build (long) ---"
python3 setup.py bdist_wheel > /tmp/fa2build.log 2>&1
STATUS=$?
echo "=== build exit: $STATUS ==="
if [ $STATUS -ne 0 ]; then
  echo "--- error lines ---"
  grep -n -E "error:|fatal error|Killed|No such file|out of memory" /tmp/fa2build.log | head -25
  echo "--- last 40 lines ---"
  tail -40 /tmp/fa2build.log
  exit $STATUS
fi
echo "--- dist ---"
ls -la dist/
echo "--- install ---"
pip install --no-cache-dir --force-reinstall --no-deps dist/*.whl 2>&1 | tail -3
echo "--- verify ---"
cd /tmp || exit 1
python3 -c "
import flash_attn, torch
print('flash_attn', flash_attn.__version__)
print('has varlen', hasattr(flash_attn, 'flash_attn_varlen_func'))
from flash_attn import flash_attn_varlen_func
q = torch.randn(4, 8, 64, dtype=torch.float16, device='cuda')
cu = torch.tensor([0, 4], dtype=torch.int32, device='cuda')
out = flash_attn_varlen_func(q, q, q, cu, cu, 4, 4)
print('smoke test output', tuple(out.shape), out.dtype)
"
