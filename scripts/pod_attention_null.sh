#!/bin/bash
# Null baseline for the attention contrast: 30 vs 40 steps, both of which were stable in
# the colour sweeps. If this contrast is as large as 20 vs 30, the 20-vs-30 result
# measures "different step counts" rather than drift.
#
# Ovi's UniPC grid is a shifted linspace, so 40 steps at indices 0,4,...,36 lands on the
# same ten timesteps as 20 steps at 0,2,...,18 and 30 steps at 0,3,...,27.
set -uo pipefail

source /local/ovi-env/bin/activate
export PYTHONPATH=/local/wsp/src:/local/Ovi
cd /local/Ovi

while pgrep -f pod_residual_v2.sh > /dev/null; do sleep 30; done
echo "=== residual chain finished; starting null baseline $(date -u +%H:%M:%S) ==="

python3 -u /local/wsp/scripts/pod_ovi_attention.py \
  --config /local/Ovi/ovi/configs/inference/inference_fusion_wsp.yaml \
  --prompt "A red fox trotting through fresh snow, wind and soft footsteps" \
  --seed 42 \
  --rollouts 30 40 \
  --capture-steps-for 30=0,3,6,9,12,15,18,21,24,27 40=0,4,8,12,16,20,24,28,32,36 \
  --layers 0 3 6 9 12 15 18 21 24 27 \
  --streams video audio \
  --cfg-pass cond \
  --max-gap 1.0 \
  --chunk 512 \
  --height-width 960x960 \
  --out /local/captures/attention_30v40_null.json
echo "=== exit: $? ==="