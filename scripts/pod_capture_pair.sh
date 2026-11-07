#!/bin/bash
# The drifting-vs-stable capture pair: the contrast the WSP direction is derived from.
#
# 20 steps drifted colour in 2 of 3 seeds in the earlier sweeps; 30 steps was stable
# in all 3. Same prompt, same seed, so step count is the only difference.
#
# Layers: 0, 2, 4, 8 and 11. Block 11 is the default slg_layer, and Ovi skips it in the
# negative pass only -- capturing it would give a conditional-only state. It is
# included deliberately so the asymmetry is visible and can be excluded downstream,
# rather than silently absent.
set -uo pipefail

source /local/ovi-env/bin/activate
cd /local/Ovi || exit 1
export PYTHONPATH="/local/wsp/src:/local/Ovi:${PYTHONPATH:-}"
export OVI_ROOT=/local/Ovi

# The repo is at /local/Ovi (capital O); the config lives inside it. An earlier version
# of this script pointed at /local/ovi, which is a different directory that only ever
# held a stray configs/ tree, so the load failed with FileNotFoundError.
CFG=/local/Ovi/ovi/configs/inference/inference_fusion_wsp.yaml
PROMPT="A medium shot of a coastal road at golden hour, waves breaking against rocks, engines and gulls audible"
# Written to the container disk, NOT /workspace. The network volume reports 461TB free
# but enforces a per-volume quota, and it is full: writes there fail with
# "OSError: [Errno 122] Disk quota exceeded" after exactly 640MB. That is also what
# silently killed the earlier detached runs -- the failure surfaced only once the job was
# run attached.
OUT="${1:-/local/captures/pair_20v30.npz}"

echo "--- backend gate ---"
python3 -c "
from wsp.hooks import attention_backend
b = attention_backend()
print('backend:', b)
raise SystemExit(0 if b == 'flash_attn_2' else 1)
" || { echo "ABORT: not FA2"; exit 1; }

mkdir -p /local/captures
rm -f /workspace/captures/pair_20v30.npz
echo "--- disk before ---"
df -h / | tail -1
echo "--- capturing 20 vs 30 steps ---"
python3 -u /local/wsp/scripts/pod_ovi_capture.py \
  --config "$CFG" \
  --prompt "$PROMPT" \
  --seed 42 \
  --rollouts 20 30 \
  --capture-steps 0 2 4 6 8 10 12 14 16 18 \
  --layers 0 2 4 8 11 \
  --height-width 960x960 \
  --patch-stride 8 \
  --n-frames 121 \
  --out "$OUT"
echo "=== exit: $? ==="
df -h / | tail -1
ls -la /local/captures/
