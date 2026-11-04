#!/usr/bin/env bash
# Report vLLM install progress in the clean venv.
set -u

echo "=== install log tail ==="
tail -4 /workspace/vllm2.log 2>/dev/null || echo "(no log)"

echo "=== import check ==="
/workspace/venv2/bin/python -c 'import vllm; print("vllm", vllm.__version__)' 2>&1 | tail -3
