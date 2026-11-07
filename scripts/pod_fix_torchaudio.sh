#!/bin/bash
set -uo pipefail
source /local/ovi-env/bin/activate
echo "--- pinning torchaudio / torchvision to torch 2.6.0 ---"
pip install --no-cache-dir "torchaudio==2.6.0" "torchvision==0.21.0" 2>&1 | tail -4
echo "--- verify ---"
python3 -c "import torch, torchaudio, torchvision; print('torch', torch.__version__, 'audio', torchaudio.__version__, 'vision', torchvision.__version__)" 2>&1 | tail -4
