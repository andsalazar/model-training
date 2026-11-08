#!/bin/bash
# Run the attention-map contrast on the pod: drifting (20 steps) vs stable (30 steps).
#
# This is the measurement the residual projection could not make. The residual direction
# aligned weakly with the two matrices that write to the stream (best 0.129 against a
# 0.018 chance level), which is what a sum-of-components direction would look like if the
# distinguishing signal lived in the attention pattern instead. This run asks directly.
#
# The capture is small: a summary is a few hundred floats per (layer, step), so the whole
# run writes a JSON of a few MB. The full attention matrix is never materialised.
#
# v2: calls are labelled by (stream, block) from self_attn hooks, the conditional CFG pass
# is recorded, and both runs capture the same ten UniPC timesteps so every pair is exact.
set -uo pipefail

source /local/ovi-env/bin/activate
export PYTHONPATH=/local/wsp/src:/local/Ovi

# ovi.ovi_fusion_engine loads 'ovi/configs/inference/inference_fusion.yaml' relative to the
# working directory at import time. Importing it from anywhere else raises
# FileNotFoundError before any of this script's own arguments are read.
cd /local/Ovi

CFG=/local/Ovi/ovi/configs/inference/inference_fusion_wsp.yaml
OUT=/local/captures/attention_20v30_v2.json
PROMPT="A red fox trotting through fresh snow, wind and soft footsteps"

if [ ! -f "$CFG" ]; then
  echo "ABORT: config not found at $CFG"
  exit 1
fi

mkdir -p /local/captures

echo "=== attention contrast: 20 vs 30 steps ==="
python3 -u /local/wsp/scripts/pod_ovi_attention.py \
  --config "$CFG" \
  --prompt "$PROMPT" \
  --seed 42 \
  --rollouts 20 30 \
  --capture-steps-for 20=0,2,4,6,8,10,12,14,16,18 30=0,3,6,9,12,15,18,21,24,27 \
  --layers 0 3 6 9 12 15 18 21 24 27 \
  --streams video audio \
  --cfg-pass cond \
  --max-gap 1.0 \
  --chunk 512 \
  --height-width 960x960 \
  --out "$OUT"
echo "=== exit: $? ==="
ls -la "$OUT" 2>/dev/null
