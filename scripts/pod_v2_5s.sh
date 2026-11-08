#!/bin/bash
# (1) effective-resolution check, including a raw VAE decode of the saved 10s latents
# (2) Stage 7 v2 at 5s: teacher (reusing the pilot's 9) -> train (3 epochs) -> eval
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi OVI_ROOT=/local/Ovi
while pgrep -f pod_run_check.sh > /dev/null; do sleep 10; done

echo "=== resolution $(date -u +%H:%M:%S) ==="
E=/local/pilot5s/eval
python3 -u /local/wsp/scripts/pod_effective_resolution.py \
  /local/long30/10s_30steps_s100.mp4 /local/long30/30s_30steps_s100.mp4 \
  /local/pilot5s/teacher/river_s100.mp4 $E/base_unipc30_s100.mp4 $E/base_unipc12_s100.mp4 \
  $E/base_euler12_s100.mp4 $E/adapter_euler12_s100.mp4 \
  --latents /local/long30/10s_30steps_s100_latents.pt --latents-mp4 /local/long30/10s_30steps_s100.mp4 \
  --model-name 960x960_10s --out /local/captures/resolution.json
echo "=== resolution exit $? ==="

W=/local/adapter5s_v2
R="python3 -u /local/wsp/scripts/pod_adapter_run.py"
C="--set v2_5s --work $W --teacher-dir /local/pilot5s/teacher"
for stage in teacher train eval; do
  echo "=== $stage $(date -u +%H:%M:%S) ==="
  if [ "$stage" = train ]; then
    $R train $C --epochs 3 --lr 1e-3 --lr-min 1e-4 --cfg-grad joint || { echo "ABORT train"; exit 1; }
  else
    $R $stage $C || { echo "ABORT $stage"; exit 1; }
  fi
done
echo "=== v2 5s done $(date -u +%H:%M:%S) ==="