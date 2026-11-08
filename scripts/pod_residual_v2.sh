#!/bin/bash
# Re-run the residual pipeline with the review fixes: conditional CFG pass, the config's
# sampler settings, and timestep-matched capture steps. Waits for any running attention
# capture first, because both need the whole GPU.
#
#   capture pair -> derive video + audio directions -> project both onto o / ffn.2
set -uo pipefail

source /local/ovi-env/bin/activate
cd /local/Ovi || exit 1
export PYTHONPATH="/local/wsp/src:/local/Ovi:${PYTHONPATH:-}"

while pgrep -f pod_ovi_attention.py > /dev/null; do sleep 30; done
echo "=== attention run finished; starting residual v2 $(date -u +%H:%M:%S) ==="

PAIR=/local/captures/pair_20v30_v2.npz
WEIGHTS=/local/Ovi/ckpts/Ovi/model_960x960.safetensors

bash /local/wsp/scripts/pod_capture_pair.sh "$PAIR" || { echo "ABORT: capture failed"; exit 1; }

for STREAM in video audio; do
  echo "=== derive $STREAM ==="
  python3 -u /local/wsp/scripts/pod_derive_direction.py \
    --capture "$PAIR" \
    --out "/local/captures/direction_${STREAM}_v2.json" \
    --steps-a 20 --steps-b 30 \
    --stream "$STREAM" \
    --n-frames 121 \
    --keep-vectors || { echo "derive $STREAM failed"; continue; }
  echo "=== project $STREAM ==="
  python3 -u /local/wsp/scripts/pod_project_direction.py \
    --weights "$WEIGHTS" \
    --report "/local/captures/direction_${STREAM}_v2.json" \
    --out "/local/captures/projection_${STREAM}_v2.json" || echo "project $STREAM failed"
done
echo "=== residual v2 done $(date -u +%H:%M:%S) ==="
ls -la /local/captures/
