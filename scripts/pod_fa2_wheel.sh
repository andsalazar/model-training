#!/bin/bash
# Try prebuilt FlashAttention wheels before falling back to another source build.
#
# The source build compiles every kernel for five architectures because
# flash-attention's setup.py reads FLASH_ATTN_CUDA_ARCHS and does not honour
# TORCH_CUDA_ARCH_LIST; on this host that means five full compiles. A prebuilt wheel
# for torch 2.6 / cu12 / cp312 would avoid the whole problem.
#
# Ovi needs flash_attn.flash_attn_varlen_func with window_size, which is standard.
set -uo pipefail

source /local/ovi-env/bin/activate
echo "--- torch/python ---"
python3 -c "import torch, sys; print('torch', torch.__version__, 'py', sys.version.split()[0])"

echo "--- check if a build is already running ---"
pgrep -f 'setup.py bdist' >/dev/null && echo "BUILD STILL RUNNING" || echo "no build running"

TRY=(
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)

for url in "${TRY[@]}"; do
  echo "--- trying $url ---"
  if pip install --no-cache-dir --no-deps "$url" 2>&1 | tail -3; then
    if python3 -c "
import torch
from flash_attn import flash_attn_varlen_func
q = torch.randn(4, 8, 64, dtype=torch.float16, device='cuda')
cu = torch.tensor([0, 4], dtype=torch.int32, device='cuda')
out = flash_attn_varlen_func(q, q, q, cu, cu, 4, 4)
print('SMOKE OK', tuple(out.shape), out.dtype)
" 2>&1 | tail -4; then
      echo "=== wheel worked ==="
      python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
      exit 0
    fi
  fi
done

echo "=== no prebuilt wheel worked; source build is required ==="
exit 1
