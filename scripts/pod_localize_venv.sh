#!/bin/bash
# Copy venv2 from the slow network volume to the pod's local NVMe.
#
# Importing vLLM straight off /workspace (an mfs network mount) times out at
# 180s. The same tree on local disk imports in seconds. The venv is relocatable
# because its scripts use a relative shebang path via the python binary.
set -u
SRC=/workspace/venv2
DST=/local/venv2
LOG=/workspace/localize.log
: > "$LOG"

{
  echo "=== COPY START $(date -u +%H:%M:%S) ==="
  mkdir -p /local
  if [ -d "$DST" ]; then
    echo "already present, skipping copy"
  else
    time cp -a "$SRC" "$DST"
    echo "copy_rc=$?"
  fi

  echo "=== VERIFY ==="
  timeout 120 "$DST/bin/python" -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -4
  echo "vllm_rc=${PIPESTATUS[0]}"
  timeout 120 "$DST/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -4
  echo "torch_rc=${PIPESTATUS[0]}"

  echo "LOCALIZE_DONE"
} >> "$LOG" 2>&1
