#!/bin/bash
# Smoke test of the Stage 7 pilot plumbing: every stage, minutes not hours.
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi OVI_ROOT=/local/Ovi PILOT_SMOKE=1
rm -rf /local/pilot_smoke
for stage in teacher train eval; do
  echo "=== $stage $(date -u +%H:%M:%S) ==="
  python3 -u /local/wsp/scripts/pod_adapter_pilot.py "$stage" --work /local/pilot_smoke --epochs 1 || { echo "ABORT: $stage"; exit 1; }
done
echo "=== smoke ok $(date -u +%H:%M:%S) ==="