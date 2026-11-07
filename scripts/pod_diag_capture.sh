#!/bin/bash
# Diagnose why the pair capture died with no output.
#
# The first attempt ran under `time ... | tail -80`, so the pipeline's exit status was
# tail's and any traceback was buffered into a file that ended up 76 bytes long. Run the
# same command without the pipe, and with the engine's own logging left at WARNING so a
# CUDA OOM is visible.
set -uo pipefail

source /local/ovi-env/bin/activate
cd /local/Ovi || exit 1
export PYTHONPATH="/local/wsp/src:/local/Ovi:${PYTHONPATH:-}"
export OVI_ROOT=/local/Ovi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CFG=/local/ovi/configs/inference_fusion_wsp.yaml
PROMPT="A medium shot of a coastal road at golden hour, waves breaking against rocks, engines and gulls audible"

# Single rollout, reduced scope: does capture work at 960x960 with a real step count?
rm -f /workspace/captures/diag.npz
python3 -u /local/wsp/scripts/pod_ovi_capture.py \
  --config "$CFG" \
  --prompt "$PROMPT" \
  --seed 42 \
  --rollouts 20 \
  --capture-steps 0 5 10 15 \
  --layers 0 2 \
  --height-width 960x960 \
  --patch-stride 8 \
  --n-frames 121 \
  --out /workspace/captures/diag.npz
echo "=== exit: $? ==="
ls -la /workspace/captures/
