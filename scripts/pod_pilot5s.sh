#!/bin/bash
# Stage 7 5s pilot: teacher -> train -> eval -> colour/audio checks.
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi OVI_ROOT=/local/Ovi
W=/local/pilot5s
for stage in teacher train eval; do
  echo "=== $stage $(date -u +%H:%M:%S) ==="
  python3 -u /local/wsp/scripts/pod_adapter_pilot.py $stage --work $W || { echo "ABORT: $stage failed"; exit 1; }
done
VIDEOS="$(ls $W/eval/*.mp4) $(ls $W/teacher/river_s{100,200,300}.mp4)"
python3 /local/wsp/scripts/pod_ovi_colour_check.py $VIDEOS --out $W/colour.json
python3 /local/wsp/scripts/pod_ovi_audio_check.py $VIDEOS --out $W/audio.json
echo "=== pilot done $(date -u +%H:%M:%S) ==="