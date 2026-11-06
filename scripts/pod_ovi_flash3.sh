#!/usr/bin/env bash
# Build FlashAttention from source with full error capture.
#
# The previous attempts swallowed the nvcc error: pip's tail only showed the Python
# traceback, not the compiler output. This runs setup.py directly and captures
# everything so the real failure is visible.
set -uo pipefail

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
# MAX_JOBS=16 got the build OOM-killed: each nvcc process compiling a flash-attention
# translation unit peaks at several GB, and 16 at once exhausted the container's memory
# limit. The log showed 24 bare "Killed" lines with no compiler error, which is the
# signature of the OOM killer rather than a code problem. 4 is slow but survives.
export MAX_JOBS=4

# shellcheck disable=SC1091
source /workspace/ovi-env/bin/activate

echo "=== toolchain ==="
which python python3 nvcc ninja
python --version
nvcc --version | tail -2

echo "=== clean previous build ==="
cd /workspace/flash-attention
rm -rf build dist *.egg-info 2>/dev/null || true

echo "=== build ==="
python setup.py build > /workspace/fbuild.log 2>&1
STATUS=$?
echo "build exit: $STATUS"

if [ "$STATUS" -ne 0 ]; then
  echo "=== compiler errors ==="
  grep -iE 'error|fatal|Killed|out of memory' /workspace/fbuild.log | head -30
  echo "=== last 40 lines ==="
  tail -40 /workspace/fbuild.log
  exit "$STATUS"
fi

echo "=== install ==="
python setup.py install 2>&1 | tail -5

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
