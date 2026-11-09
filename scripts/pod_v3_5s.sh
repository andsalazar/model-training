#!/bin/bash
# Stage 7 v3 at 5 s: broader eval + on-policy training (parts 1 and 3) + 0-3 placement.
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi OVI_ROOT=/local/Ovi
W=/local/v3_5s
R="python3 -u /local/wsp/scripts/pod_adapter_run.py"
C="--set v3_5s"
step() { echo "=== $1 $(date -u +%H:%M:%S) ==="; shift; "$@" || { echo "ABORT $*"; exit 1; }; }
step teacher $R teacher $C --work $W --only eval
step onpolicy $R onpolicy $C --work $W --init-adapter /local/adapters/v2_5s.pt \
  --epochs 2 --lr 3e-4 --lr-min 3e-5 --substeps 4 --save-data
step onpolicy_b03 $R onpolicy $C --work $W/b03 --blocks 0 1 2 3 --init-adapter /local/adapters/v2_5s.pt \
  --epochs 1 --lr 3e-4 --lr-min 3e-5 --substeps 4 --data-from $W/onpolicy/data/epoch0
step eval $R eval $C --work $W --variants base_unipc12 base_unipc30 \
  --adapters v2=/local/adapters/v2_5s.pt v2fix=/local/adapters/v2_5s.pt@audio_off=0,1 \
             onpol=$W/onpolicy/adapter_best.pt onpol03=$W/b03/onpolicy/adapter_best.pt@blocks=0,1,2,3
step adherence python3 -u /local/wsp/scripts/pod_adherence.py --eval-dir $W/eval --teacher-dir $W/teacher --out $W/adherence.json
echo "=== v3 5s done $(date -u +%H:%M:%S) ==="