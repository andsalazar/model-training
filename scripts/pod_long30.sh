#!/bin/bash
# 30s probe (method A): the 10s model run at 3x its trained length, plus a 10s baseline
# measured with the same tool. Waits for the 5s pilot because both need the whole GPU.
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi OVI_ROOT=/local/Ovi
while pgrep -f pod_pilot5s.sh > /dev/null; do sleep 30; done
for secs in 10 30; do
  echo "=== ${secs}s $(date -u +%H:%M:%S) ==="
  python3 -u /local/wsp/scripts/pod_ovi_long.py --seconds "$secs" --steps 30 --seed 100 --out-dir /local/long30
  echo "=== ${secs}s exit $? $(date -u +%H:%M:%S) ==="
done