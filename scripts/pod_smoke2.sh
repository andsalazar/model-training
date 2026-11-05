#!/bin/bash
# Retry the vLLM load and capture the actual root-cause error.
set -u
LOG=/workspace/smoke2.log
: > "$LOG"

{
  echo "=== GPU ==="
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader 2>&1
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>&1

  echo "=== VLLM LOAD (verbose) ==="
  timeout 600 /local/venv/bin/python - <<'EOF' 2>&1 | grep -viE 'warning|deprecat' | tail -40
from vllm import LLM, SamplingParams
llm = LLM(model="/local/models/qwen", max_model_len=4096, gpu_memory_utilization=0.85, enforce_eager=True)
out = llm.generate(["What is 12*11? Answer with just the number."], SamplingParams(max_tokens=32, temperature=0))
print("GEN:", repr(out[0].outputs[0].text))
print("VLLM_LOAD_OK")
EOF
  echo "load_rc=$?"

  echo "SMOKE2_DONE"
} >> "$LOG" 2>&1
