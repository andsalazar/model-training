#!/bin/bash
# v3 smoke test: every new path in minutes (numbers mean nothing). Also times the H100.
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi OVI_ROOT=/local/Ovi
W=/local/v3_check
R="python3 -u /local/wsp/scripts/pod_adapter_run.py"
step() { echo "=== $1 $(date -u +%H:%M:%S) ==="; shift; "$@" || { echo "ABORT $*"; exit 1; }; }
step teacher $R teacher --set v3_check --work $W --only eval
step onpolicy $R onpolicy --set v3_check --work $W --init-adapter /local/adapters/v2_5s.pt \
  --epochs 1 --max-items 1 --max-samples 2 --save-data
step onpolicy_b03 $R onpolicy --set v3_check --work $W/b03 --blocks 0 1 2 3 \
  --init-adapter /local/adapters/v2_5s.pt --epochs 1 --max-items 1 --max-samples 2 \
  --data-from $W/onpolicy/data/epoch0
step eval $R eval --set v3_check --work $W --variants base_unipc12 \
  --adapters v2fix=/local/adapters/v2_5s.pt@audio_off=0,1 onpol03=$W/b03/onpolicy/adapter_best.pt@blocks=0,1,2,3
step adherence python3 -u /local/wsp/scripts/pod_adherence.py --eval-dir $W/eval --teacher-dir $W/teacher --out $W/adherence.json
echo "=== smoke ok $(date -u +%H:%M:%S) ==="