#!/usr/bin/env bash
# Report vLLM install progress and whether it imports.
set -u

echo "=== install log tail ==="
tail -6 /workspace/vllm_install.log 2>/dev/null || echo "(no log)"

echo "=== import check ==="
/workspace/venv/bin/python -c 'import vllm; print("vllm", vllm.__version__)' 2>&1 | tail -3
