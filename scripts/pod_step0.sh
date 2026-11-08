#!/bin/bash
# Step 0 of the adapter-first plan:
#   (a) adapter training memory/time probe (5s weights, 5s and 10s token counts)
#   (b) 10s quality-loss curve: 16 and 12 steps at seeds 100/200/300, plus a 20-step
#       seed-100 anchor that should reproduce the earlier pod's drift (R^2 0.589)
# Earlier 10s results at 50/30/20 steps are in docs/stage6-speed-findings.md.
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi
export OVI_ROOT=/local/Ovi WSP_SRC=/local/wsp/src

echo "=== (a) adapter probe $(date -u +%H:%M:%S) ==="
python3 -u /local/wsp/scripts/pod_adapter_probe.py \
  --config /local/Ovi/ovi/configs/inference/inference_fusion_wsp.yaml \
  --out /local/captures/adapter_probe.json
echo "=== probe exit: $? ==="

while pgrep -f pod_fetch_10s.sh > /dev/null; do sleep 20; done
ls -la /local/Ovi/ckpts/Ovi/model_960x960_10s.safetensors || { echo "ABORT: no 10s weights"; exit 1; }

echo "=== (b) 10s sweep $(date -u +%H:%M:%S) ==="
python3 -u /local/wsp/scripts/pod_ovi_speed.py \
  --horizons 960x960_10s --steps 16 12 --seeds 100 200 300 \
  --work /local/ovi_speed10 --out /local/captures/speed10_low.json
python3 -u /local/wsp/scripts/pod_ovi_speed.py \
  --horizons 960x960_10s --steps 20 --seeds 100 \
  --work /local/ovi_speed10_anchor --out /local/captures/speed10_anchor.json

VIDEOS=$(find /local/ovi_speed10 /local/ovi_speed10_anchor -name '*.mp4' | sort)
echo "$VIDEOS"
python3 /local/wsp/scripts/pod_ovi_colour_check.py $VIDEOS --out /local/captures/speed10_colour.json
python3 /local/wsp/scripts/pod_ovi_audio_check.py $VIDEOS --out /local/captures/speed10_audio.json
echo "=== step 0 done $(date -u +%H:%M:%S) ==="