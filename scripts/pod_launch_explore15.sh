#!/bin/bash
# explore15: run the editor with the verification harness and re-measure its edits.
set -u
cd /workspace || exit 1

export PATH=/local/venv/bin:$PATH
export PYTHONPATH=/workspace/src
export TOKENIZERS_PARALLELISM=false

CKPT=/local/models/qwen/model.safetensors-00001-of-00001.safetensors
OUT=/workspace/explore15.json

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY not set" >&2
  exit 1
fi

/local/venv/bin/python /workspace/scripts/run_explore_verified.py \
  --editor deepseek \
  --model /local/models/qwen \
  --checkpoint "$CKPT" \
  --out "$OUT" \
  --max-turns 16 \
  --max-forward-passes 26 \
  --max-tokens 2048 \
  2>&1 | tee /workspace/explore15.log

echo "RUN_EXIT=$?"
