#!/bin/bash
# WSP second stage: attention-vs-teacher contrasts and adapter write maps.
# Runs after the 10s pipeline (both need the whole GPU).
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi OVI_ROOT=/local/Ovi
while pgrep -f pod_v2_10s.sh > /dev/null; do sleep 60; done
mkdir -p /local/wsp_stage
echo "=== 5s $(date -u +%H:%M:%S) ==="
python3 -u /local/wsp/scripts/pod_wsp_adapter.py --adapter /local/adapter5s_v2/train/adapter_best.pt \
  --tags violin_100 river_100 river_200 --out /local/wsp_stage/wsp_5s.json
echo "=== 5s exit $? ==="
A10=/local/adapter10s_v2/train/adapter_best.pt
if [ -f "$A10" ]; then
  echo "=== 10s $(date -u +%H:%M:%S) ==="
  python3 -u /local/wsp/scripts/pod_wsp_adapter.py --model-name 960x960_10s --adapter "$A10" \
    --tags river_100 --out /local/wsp_stage/wsp_10s.json
  echo "=== 10s exit $? ==="
fi
echo "=== wsp stage done $(date -u +%H:%M:%S) ==="