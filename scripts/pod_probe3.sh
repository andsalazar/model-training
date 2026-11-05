#!/bin/bash
# Probe the pod without blocking SSH: everything goes to a log file.
set -u
PY=/workspace/venv2/bin/python
LOG=/workspace/probe3.log
: > "$LOG"

{
  echo "=== PYTHON ==="
  "$PY" -V 2>&1

  echo "=== VLLM ==="
  timeout 120 "$PY" -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -3
  echo "vllm_exit=$?"

  echo "=== TORCH ==="
  timeout 120 "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -3
  echo "torch_exit=$?"

  echo "=== MODEL DIRS ==="
  for d in /workspace/qwen /workspace/models /workspace/model /root/qwen; do
    if [ -d "$d" ]; then echo "FOUND $d"; ls "$d" | head -6; fi
  done
  echo "safetensors:"
  ls /workspace/*/*.safetensors 2>/dev/null | head -5
  echo "hf cache:"
  ls /root/.cache/huggingface/hub 2>/dev/null | head -5

  echo "=== DISK ==="
  df -h /workspace 2>&1 | tail -1

  echo "=== GPU ==="
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1

  echo "PROBE_DONE"
} >> "$LOG" 2>&1
