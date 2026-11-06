#!/usr/bin/env bash
# Rebuild and install FlashAttention, saving the wheel so it survives a mistake.
#
# The previous build's wheel was deleted by `pip install` cleaning its cache, and the
# installed package was then removed by mistake when clearing the SDPA shim (the shim
# had overwritten the same directory). Rebuilding takes ~90 minutes, so this time the
# wheel is copied to /workspace/wheels first and installed from there.
set -euo pipefail

source /workspace/ovi-env/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4

mkdir -p /workspace/wheels

cd /workspace/flash-attention
# The build directory already has all 96 compiled objects, so this should only relink.
echo "=== building wheel (objects already compiled) ==="
python setup.py bdist_wheel 2>&1 | tail -12

echo "=== saving wheel ==="
WHEEL=$(ls -t dist/*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
  echo "NO WHEEL PRODUCED"
  exit 1
fi
cp "$WHEEL" /workspace/wheels/
ls -la /workspace/wheels/

echo "=== installing from saved wheel ==="
pip install --no-cache-dir --force-reinstall --no-deps "/workspace/wheels/$(basename "$WHEEL")" 2>&1 | tail -5

echo "=== verify ==="
python - <<'PY'
import torch
import flash_attn
print("flash_attn", flash_attn.__version__)
print("file", flash_attn.__file__)
q = torch.randn(2, 64, 8, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn.flash_attn_func(q, q, q)
torch.cuda.synchronize()
print("kernel OK", tuple(out.shape))
print("varlen:", hasattr(flash_attn, "flash_attn_varlen_func"))
PY
echo "=== done ==="
