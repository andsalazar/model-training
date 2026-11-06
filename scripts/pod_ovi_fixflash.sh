#!/usr/bin/env bash
# Remove the flash_attn build that was compiled against a different torch ABI.
#
# The pod's system Python ships torch 2.8.0; the venv installs 2.6.0. flash_attn was
# built against 2.8.0's C++ ABI, so importing it under 2.6.0 fails with an undefined
# symbol for c10::Error. FlashAttention is optional for Ovi â€” it only affects speed â€”
# so removing it is the correct fix rather than rebuilding against 2.6.0.
set -uo pipefail

source /workspace/ovi-env/bin/activate

echo "=== uninstall flash_attn ==="
pip uninstall -y flash_attn flash-attn 2>&1 | tail -5

echo "=== verify imports ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
from diffusers import FluxPipeline
print("diffusers OK")
import transformers
print("transformers", transformers.__version__)
PY
