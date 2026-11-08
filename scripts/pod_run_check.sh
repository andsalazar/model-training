#!/bin/bash
# Plumbing check for pod_adapter_run.py on the pilot's teacher files: the sequential
# CFG gradient, text offload, best-epoch selection, and a one-variant eval.
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi OVI_ROOT=/local/Ovi
W=/local/run_check
rm -rf $W
R="python3 -u /local/wsp/scripts/pod_adapter_run.py"
$R train --set check --work $W --teacher-dir /local/pilot5s/teacher --epochs 1 --max-samples 2 \
  --cfg-grad sequential --offload-text || { echo "ABORT train"; exit 1; }
$R eval --set check --work $W --teacher-dir /local/pilot5s/teacher --variants adapter_euler12 || { echo "ABORT eval"; exit 1; }
echo "=== check ok ==="