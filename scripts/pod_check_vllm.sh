#!/bin/bash
# Check vLLM/torch importability with real exit codes, written to a log.
set -u
PY=/workspace/venv2/bin/python
LOG=/workspace/vllm_check.log
: > "$LOG"

{
  echo "=== VLLM ==="
  timeout 180 "$PY" -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -5
  echo "vllm_rc=${PIPESTATUS[0]}"

  echo "=== TORCH ==="
  timeout 180 "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -5
  echo "torch_rc=${PIPESTATUS[0]}"

  echo "=== TRANSFORMERS ==="
  timeout 120 "$PY" -c "import transformers; print('transformers', transformers.__version__)" 2>&1 | tail -3
  echo "tf_rc=${PIPESTATUS[0]}"

  echo "=== HF CLI ==="
  which hf huggingface-cli 2>&1 | head -2

  echo "CHECK_DONE"
} >> "$LOG" 2>&1
