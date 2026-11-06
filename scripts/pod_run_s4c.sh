#!/bin/bash
# Stage 4c: retrain with targets that close the thinking block and emit the answer.
#
# Stage 4b's targets were cut immediately after the answer, so the adapter learned
# "stop right after computing" and never learned to leave the reasoning block.
# Re-judged strictly, zero targets delivered a finished answer. This run appends
# the closing marker and the bare answer to each target.
set -u
cd /workspace || exit 1
export PATH=/local/venv/bin:$PATH
export PYTHONPATH=/workspace/src
export TOKENIZERS_PARALLELISM=false

LOG=/workspace/s4c_run.log
: > "$LOG"

{
  echo "=== TARGETS (close variant) ==="
  /local/venv/bin/python /workspace/scripts/pod_stage4b_targets.py \
    --close /workspace/s4c_targets.jsonl 2>&1 | grep -aE "targets [0-9]+ mean|guards [0-9]+ mean|written|SKIP"
  echo "targets_rc=$?"

  echo "=== TRAIN ==="
  /local/venv/bin/python /workspace/scripts/pod_train_lora_b.py \
    --model /local/models/qwen \
    --targets /workspace/s4c_targets.jsonl \
    --out /workspace/s4c_adapter \
    --epochs 8 2>&1 | grep -aE "epoch |max .lora|saved adapter"
  echo "train_rc=$?"

  echo "=== EVAL ==="
  /local/venv/bin/python /workspace/scripts/pod_eval_adapter.py \
    /workspace/s4c_adapter /workspace/s4c_eval.json 2>&1 | grep -aE "^  (base|with_adapter) |status:|GUARD HARM|^==="
  echo "eval_rc=$?"

  echo "S4C_DONE"
} >> "$LOG" 2>&1
