#!/bin/bash
# Measure baseline and edited checkpoints back to back on the same GPU.
set -u
cd /workspace || exit 1

export PATH=/workspace/venv2/bin:$PATH
export PYTHONPATH=/workspace/src
export TOKENIZERS_PARALLELISM=false

BASE=/workspace/qwen
EDIT=/workspace/edit_L12_a

echo "########## BASELINE ##########"
/workspace/venv2/bin/python /workspace/scripts/pod_verify_edit13.py "$BASE" /workspace/verify13_base.json 2>&1 | grep -vE 'WARNING|INFO|Fetching|Loading|Processed|Rendering|Parse|torch\._'

echo "########## EDITED ##########"
/workspace/venv2/bin/python /workspace/scripts/pod_verify_edit13.py "$EDIT" /workspace/verify13_edit.json 2>&1 | grep -vE 'WARNING|INFO|Fetching|Loading|Processed|Rendering|Parse|torch\._'

echo "DONE"
