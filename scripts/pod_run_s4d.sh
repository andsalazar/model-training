#!/bin/bash
# Stage 4d: train on targets ONLY, to test whether the guard damage came from the
# mixed training set or from the mechanism itself.
#
# 4c fixed the targets (6/6 delivered) but pushed guard_hammer and guard_blue_sky
# to the token cap. The mixed set taught one adapter to both suppress deliberation
# (targets) and preserve it (guards). If a target-only adapter fixes targets
# without touching guards, the mixed set was the problem. If it still damages
# guards, the mechanism is genuinely non-selective.
set -u
cd /workspace || exit 1
export PATH=/local/venv/bin:$PATH
export PYTHONPATH=/workspace/src
export TOKENIZERS_PARALLELISM=false

LOG=/workspace/s4d_run.log
: > "$LOG"

{
  echo "=== TRAIN (targets only) ==="
  /local/venv/bin/python /workspace/scripts/pod_train_lora_b.py \
    --model /local/models/qwen \
    --targets /workspace/s4c_targets.jsonl \
    --kinds target \
    --out /workspace/s4d_adapter \
    --epochs 8 2>&1 | grep -aE "train [0-9]+|epoch |max .lora|saved adapter"
  echo "train_rc=$?"

  echo "=== EVAL ==="
  /local/venv/bin/python /workspace/scripts/pod_eval_adapter.py \
    /workspace/s4d_adapter /workspace/s4d_eval.json 2>&1 | grep -aE "^  (base|with_adapter) |status:|GUARD HARM|^==="
  echo "eval_rc=$?"

  echo "S4D_DONE"
} >> "$LOG" 2>&1
