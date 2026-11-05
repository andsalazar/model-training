#!/bin/bash
# Launcher for explore13: activation-driven edit with token_window + direction_similarity.
set -u
cd /workspace || exit 1

export PATH=/workspace/venv2/bin:$PATH
export PYTHONPATH=/workspace/src
export TOKENIZERS_PARALLELISM=false

CKPT=/workspace/qwen/model.safetensors-00001-of-00001.safetensors
OUT=/workspace/explore13.json

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY not set" >&2
  exit 1
fi

/workspace/venv2/bin/python /workspace/scripts/run_explore.py \
  --editor deepseek \
  --model Qwen/Qwen3.5-0.8B \
  --checkpoint "$CKPT" \
  --out "$OUT" \
  --max-turns 14 \
  --max-forward-passes 22 \
  --max-tokens 2048 \
  2>&1 | tee /workspace/explore13.log

echo "EXIT_STATUS=$?"
