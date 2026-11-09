#!/bin/bash
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi OVI_ROOT=/local/Ovi
python3 -u /local/wsp/scripts/pod_flash_ablation.py --adapter /local/adapter5s_v2/train/adapter_best.pt \
  --tags river_100 river_200 river_300 violin_100 --variants audio_off_first2 video_only \
  --save-dir /local/wsp_stage/fix_eval2 --out /local/wsp_stage/fix_eval2.json
CUDA_VISIBLE_DEVICES= python3 /local/wsp/scripts/pod_ovi_audio_check.py /local/wsp_stage/fix_eval2/*.mp4 --out /local/wsp_stage/fix2_audio_mp4.json > /dev/null
echo "=== fix eval2 exit $? ==="