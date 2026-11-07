#!/bin/bash
# Run the Ovi WSP capture on the pod.
#
# Requires flash_attn: Ovi's SDPA fallback sets attn_mask = None and attends over
# padding tokens, so a capture taken through it would be a different computation.
# The individual script checks and refuses rather than producing contaminated states.
#
# The drifting-against-stable pair is 20 steps versus 30 at one prompt and seed, which
# is what the earlier sweeps established: 20 steps drifted colour in 2 of 3 seeds, 30
# steps was stable.
set -uo pipefail

source /local/ovi-env/bin/activate
cd /local/Ovi || exit 1

# wsp is the local package under test; it is not pip-installed on the pod.
export PYTHONPATH="/local/wsp/src:/local/Ovi:${PYTHONPATH:-}"
export OVI_ROOT=/local/Ovi

echo "=== attention backend ==="
python3 - <<'PY'
import sys

from wsp.hooks import attention_backend

backend = attention_backend()
print("backend:", backend)
if backend != "flash_attn_2":
    sys.exit(
        f"refusing to capture: backend is {backend!r}. Ovi's SDPA fallback drops "
        "the padding mask, so the captured states would not be the states the "
        "model occupies under FlashAttention."
    )
PY
STATUS=$?
if [ $STATUS -ne 0 ]; then
  echo "ABORT: flash_attn is not available"
  exit 1
fi

echo "=== config ==="
CFG=/local/ovi/configs/inference_fusion_wsp.yaml
if [ ! -f "$CFG" ]; then
  cp /local/Ovi/ovi/configs/inference/inference_fusion.yaml "$CFG"
  # 5s checkpoint: the capture needs a forward pass, not a long video, and the 5s
  # model is the one downloaded. 960x960_10s would need another 23GB.
  sed -i 's/^model_name:.*/model_name: "960x960_5s"/' "$CFG"
  sed -i 's|^ckpt_dir:.*|ckpt_dir: /local/Ovi/ckpts|' "$CFG"
  sed -i 's/^slg_layer:.*/slg_layer: 11/' "$CFG"
fi
grep -E 'model_name|ckpt_dir|slg_layer' "$CFG"

PROMPT="${1:-A medium shot of a coastal road at golden hour, waves breaking against rocks, engines and gulls audible}"
OUT="${2:-/local/captures/wsp_capture.npz}"

echo "=== capture ==="
python3 /local/wsp/scripts/pod_ovi_capture.py \
  --config "$CFG" \
  --prompt "$PROMPT" \
  --seed 42 \
  --rollouts 20 30 \
  --capture-steps 0 1 2 3 4 5 6 7 8 9 \
  --layers 0 1 2 4 8 11 \
  --out "$OUT"
