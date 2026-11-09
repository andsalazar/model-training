#!/bin/bash
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi OVI_ROOT=/local/Ovi
python3 -u /local/wsp/scripts/pod_flash_ablation.py --adapter /local/adapter5s_v2/train/adapter_best.pt \
  --tags violin_100 river_100 --out /local/wsp_stage/flash_ablation.json
echo "=== ablation exit $? ==="