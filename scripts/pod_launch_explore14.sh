#!/bin/bash
# Launcher for explore14: the editor now has the column-list form and the
# misquoted-number auditor. The task asks for held-out measurement.
set -u
cd /workspace || exit 1

export PATH=/workspace/venv2/bin:$PATH
export PYTHONPATH=/workspace/src
export TOKENIZERS_PARALLELISM=false

CKPT=/workspace/qwen/model.safetensors-00001-of-00001.safetensors
OUT=/workspace/explore14.json

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY not set" >&2
  exit 1
fi

/workspace/venv2/bin/python /workspace/scripts/run_explore.py \
  --editor deepseek \
  --model Qwen/Qwen3.5-0.8B \
  --checkpoint "$CKPT" \
  --out "$OUT" \
  --max-turns 16 \
  --max-forward-passes 26 \
  --max-tokens 2048 \
  2>&1 | tee /workspace/explore14.log

echo "EXIT_STATUS=$?"
