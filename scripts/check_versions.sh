#!/usr/bin/env bash
# Report the versions that decide whether the optimized kernels can load.
set -u

echo "=== venv ==="
/workspace/venv/bin/pip list 2>/dev/null | grep -Ei '^(triton|torch|fla|flash|transformers) '

echo "=== system ==="
pip list 2>/dev/null | grep -Ei '^(triton|torch|transformers) '

echo "=== triton import ==="
/workspace/venv/bin/python -c 'import triton; print("triton", triton.__version__)' 2>&1 | tail -2
