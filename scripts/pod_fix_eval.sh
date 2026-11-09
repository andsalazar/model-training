#!/bin/bash
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi OVI_ROOT=/local/Ovi
python3 -u /local/wsp/scripts/pod_flash_ablation.py --adapter /local/adapter5s_v2/train/adapter_best.pt \
  --tags river_100 river_200 river_300 violin_100 --variants no_first2_steps no_first1_step \
  --save-dir /local/wsp_stage/fix_eval --out /local/wsp_stage/fix_eval.json
echo "=== fix eval exit $? ==="