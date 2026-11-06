#!/usr/bin/env bash
# Remove the SDPA shim and confirm the real FlashAttention build is what loads.
#
# The shim was installed as a fallback while FA2 compiled. Leaving it in place would
# silently substitute SDPA for the real kernel, which would make every timing number
# wrong in a way nothing else would catch.
set -euo pipefail

SHIM=/workspace/ovi-env/lib/python3.12/site-packages/flash_attn
if [ -d "$SHIM" ]; then
  echo "removing shim at $SHIM"
  rm -rf "$SHIM"
else
  echo "no shim present"
fi

source /workspace/ovi-env/bin/activate
python - <<'PY'
import torch
import flash_attn
print("flash_attn", flash_attn.__version__)
print("file", flash_attn.__file__)
assert "site-packages/flash_attn" in flash_attn.__file__, "unexpected module path"
q = torch.randn(2, 64, 8, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn.flash_attn_func(q, q, q)
torch.cuda.synchronize()
print("kernel OK", tuple(out.shape))
print("varlen available:", hasattr(flash_attn, "flash_attn_varlen_func"))
PY
