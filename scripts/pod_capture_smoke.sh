#!/bin/bash
# First real capture: a short single rollout to prove the pipeline end to end.
#
# Deliberately small -- one 8-step run, two layers, three capture steps -- so a
# failure surfaces in minutes rather than after an hour. The full drifting-vs-stable
# pair runs only once this reports a non-empty capture with FA2 as the backend.
set -uo pipefail

source /local/ovi-env/bin/activate
cd /local/Ovi || exit 1
export PYTHONPATH="/local/wsp/src:/local/Ovi:${PYTHONPATH:-}"
export OVI_ROOT=/local/Ovi

CFG=/local/ovi/configs/inference_fusion_wsp.yaml
if [ ! -f "$CFG" ]; then
  mkdir -p /local/ovi/configs
  cp /local/Ovi/ovi/configs/inference/inference_fusion.yaml "$CFG"
  sed -i 's|^model_name:.*|model_name: "960x960_5s"|' "$CFG"
  sed -i 's|^ckpt_dir:.*|ckpt_dir: /local/Ovi/ckpts|' "$CFG"
  sed -i 's|^output_dir:.*|output_dir: /local/outputs|' "$CFG"
  sed -i 's|^mode:.*|mode: "t2v"|' "$CFG"
fi
echo "--- config ---"
grep -E '^(model_name|ckpt_dir|mode|slg_layer|fp8|cpu_offload|sp_size)' "$CFG"

mkdir -p /local/captures
echo "--- capture (small) ---"
time python3 /local/wsp/scripts/pod_ovi_capture.py \
  --config "$CFG" \
  --prompt "A red fox trotting through fresh snow, wind and soft footsteps" \
  --seed 42 \
  --rollouts 8 \
  --capture-steps 0 2 4 \
  --layers 0 2 \
  --height-width 704x704 \
  --out /local/captures/smoke.npz 2>&1 | tail -60
echo "=== files ==="
ls -la /local/captures/ 2>/dev/null
