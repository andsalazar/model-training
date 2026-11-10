#!/bin/bash
# Stage 7 v4, the deciding experiment: does 12 steps + the v2 adapter beat simply running
# 16 or 20 UniPC steps? Ten unseen prompts at three seeds, every variant on this one pod.
#
# One seed at a time, teacher then variants then adherence, so a run that is cut short
# still holds all ten prompts at the seeds it finished. Every stage skips what exists, so
# the script can be re-run after an interruption.
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi OVI_ROOT=/local/Ovi
W=${1:-/local/v4_decide}
SET=${2:-decide}
SEEDS=${3:-"100 200 300"}
R="python3 -u /local/wsp/scripts/pod_adapter_run.py"
V="base_unipc12 base_unipc16 base_unipc20 base_unipc30 base_euler12"
step() { echo "=== $1 $(date -u +%H:%M:%S) ==="; shift; "$@" || { echo "ABORT $*"; exit 1; }; }
mkdir -p "$W"
rm -f "$W/DONE"
git -C /local/Ovi rev-parse HEAD > "$W/ovi_commit.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$W/gpu.txt"
for S in $SEEDS; do
  step "teacher $S" $R teacher --set "${SET}_s$S" --work "$W" --only eval
  python3 -u /local/wsp/scripts/pod_noise_hash.py --teacher-dir "$W/teacher" --out "$W/noise.json" \
    || echo "noise hash failed for seed $S"
  step "eval $S" $R eval --set "${SET}_s$S" --work "$W" --variants $V --adapters v2=/local/adapters/v2_5s.pt
  step "adherence $S" python3 -u /local/wsp/scripts/pod_adherence.py --eval-dir "$W/eval" --teacher-dir "$W/teacher" --out "$W/adherence.json"
  CUDA_VISIBLE_DEVICES= python3 /local/wsp/scripts/pod_ovi_audio_check.py "$W"/teacher/*.mp4 "$W"/eval/*.mp4 --out "$W/audio_mp4.json" > /dev/null \
    || echo "audio check failed for seed $S"
  echo "$S $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$W/seeds_done.txt"
done
date -u +%Y-%m-%dT%H:%M:%SZ > "$W/DONE"
echo "=== decide done $(date -u +%H:%M:%S) ==="
