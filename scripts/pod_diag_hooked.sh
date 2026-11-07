#!/bin/bash
# Run the actual capture attached with a modest scope, to surface the real error.
#
# Verified so far: engine loads (36.67 GB VRAM), bare generation works (55.6s for 4
# steps, returns real video and audio), engine load exit 0. Only the hooked path dies.
# Small scope so the failure appears in a couple of minutes.
set -uo pipefail

source /local/ovi-env/bin/activate
cd /local/Ovi || exit 1
export PYTHONPATH="/local/wsp/src:/local/Ovi:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== hooked capture: 4 steps, 1 layer, 2 capture steps ==="
python3 -u /local/wsp/scripts/pod_ovi_capture.py \
  --config /local/ovi/configs/inference_fusion_wsp.yaml \
  --prompt "A red fox in snow" \
  --seed 42 \
  --rollouts 4 \
  --capture-steps 0 2 \
  --layers 0 \
  --height-width 704x704 \
  --patch-stride 8 \
  --n-frames 121 \
  --out /local/captures/diag_hooked.npz
echo "=== exit: $? ==="
ls -la /local/captures/ 2>/dev/null
