#!/bin/bash
# Install Ovi's requirements.
#
# The main setup script used `set -e`, so the FlashAttention failure aborted it before
# the requirements step ever ran. flash_attn is now installed from a prebuilt wheel, so
# requirements can be installed on top. torch/torchvision/torchaudio are pinned first
# and excluded here so pip does not upgrade them to a mismatched CUDA build.
set -uo pipefail

source /local/ovi-env/bin/activate
cd /local/Ovi || exit 1

echo "=== requirements ==="
grep -vE '^(torch|torchvision|torchaudio|flash.attn)' requirements.txt > /tmp/req_no_torch.txt
echo "--- installing ---"
pip install --no-cache-dir -r /tmp/req_no_torch.txt > /tmp/req.log 2>&1
STATUS=$?
echo "exit: $STATUS"
tail -6 /tmp/req.log

echo "=== verify key imports ==="
python3 -c "
import torch, torchaudio, torchvision, einops, omegaconf, transformers, diffusers
print('torch', torch.__version__)
print('torchaudio', torchaudio.__version__)
print('einops', einops.__version__)
print('transformers', transformers.__version__)
print('diffusers', diffusers.__version__)
" 2>&1 | tail -10
