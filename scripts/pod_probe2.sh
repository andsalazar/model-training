#!/bin/bash
# Narrow probe: is vLLM importable, and where is the model?
set -u
PY=/workspace/venv2/bin/python

echo "=== PYTHON ==="
$PY -V 2>&1

echo "=== VLLM ==="
$PY -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -3

echo "=== TORCH ==="
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -3

echo "=== MODEL DIRS ==="
for d in /workspace/qwen /workspace/models /workspace/model /root/qwen; do
  if [ -d "$d" ]; then echo "FOUND $d"; ls "$d" | head -6; fi
done

echo "=== DISK ==="
df -h /workspace 2>&1 | tail -1
