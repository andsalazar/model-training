#!/bin/bash
# Stage 7 v2 at 10s: teacher -> zero-shot eval of the 5s adapter (+ base variants)
# -> train from the 5s adapter (sequential CFG gradient, T5 offloaded) -> eval trained.
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
export PYTHONPATH=/local/wsp/src:/local/Ovi OVI_ROOT=/local/Ovi
W=/local/adapter10s_v2
R="python3 -u /local/wsp/scripts/pod_adapter_run.py"
C="--set v2_10s --work $W --model-name 960x960_10s"
A5=/local/adapter5s_v2/train/adapter_best.pt
step() { echo "=== $1 $(date -u +%H:%M:%S) ==="; shift; "$@" || { echo "ABORT"; exit 1; }; }
step teacher $R teacher $C
step zeroshot $R eval $C --adapter $A5 --eval-dir $W/eval_zeroshot
step train $R train $C --init-adapter $A5 --epochs 2 --lr 5e-4 --lr-min 5e-5 --cfg-grad sequential --offload-text
step trained $R eval $C --variants adapter_euler12 --eval-dir $W/eval_trained
echo "=== v2 10s done $(date -u +%H:%M:%S) ==="