#!/bin/bash
# Rebuild Stage 4 targets and retrain, so the adapter's targets can be re-judged.
#
# The original Stage 4 adapter lived on a pod that has since been destroyed, and
# its saved eval JSON kept only text_chars, not the text. Re-judging "correct"
# needs the text, so the adapter has to be rebuilt.
set -u
cd /workspace || exit 1
export PATH=/local/venv/bin:$PATH
export PYTHONPATH=/workspace/src
export TOKENIZERS_PARALLELISM=false

LOG=/workspace/s4b_run.log
: > "$LOG"

{
  echo "=== TARGETS ==="
  /local/venv/bin/python /workspace/scripts/pod_stage4b_targets.py \
    /workspace/s4b_targets.jsonl 2>&1 | grep -aE "targets [0-9]+ mean|guards [0-9]+ mean|written|skipped"
  echo "targets_rc=$?"

  echo "=== TRAIN ==="
  /local/venv/bin/python /workspace/scripts/pod_train_lora_b.py \
    --model /local/models/qwen \
    --targets /workspace/s4b_targets.jsonl \
    --out /workspace/s4b_adapter \
    --epochs 8 2>&1 | grep -aE "epoch |max \|lora|saved adapter"
  echo "train_rc=$?"

  echo "=== EVAL ==="
  /local/venv/bin/python /workspace/scripts/pod_eval_adapter.py \
    /workspace/s4b_adapter /workspace/s4b_eval.json 2>&1 | grep -aE "^  (base|with_adapter) |status:|GUARD HARM|^==="
  echo "eval_rc=$?"

  echo "S4B_DONE"
} >> "$LOG" 2>&1
