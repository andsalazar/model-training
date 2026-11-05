#!/bin/bash
# Smoke test: does vLLM load the model and does the WSP code import?
set -u
LOG=/workspace/smoke.log
: > "$LOG"

{
  echo "=== WSP IMPORT ==="
  PYTHONPATH=/workspace/src timeout 120 /local/venv/bin/python -c \
    "import wsp.tools, wsp.verify, wsp.activations, wsp.audit; print('WSP_IMPORT_OK')" 2>&1 | tail -5

  echo "=== MODEL FILES ==="
  ls /local/models/qwen/*.safetensors 2>&1 | head -3
  cat /local/models/qwen/config.json 2>/dev/null | head -c 300
  echo

  echo "=== VLLM LOAD ==="
  timeout 600 /local/venv/bin/python - <<'EOF' 2>&1 | tail -12
from vllm import LLM, SamplingParams
llm = LLM(model="/local/models/qwen", max_model_len=4096, gpu_memory_utilization=0.85, enforce_eager=True)
out = llm.generate(["What is 12*11? Answer with just the number."], SamplingParams(max_tokens=32, temperature=0))
print("GEN:", repr(out[0].outputs[0].text))
print("VLLM_LOAD_OK")
EOF
  echo "load_rc=$?"

  echo "SMOKE_DONE"
} >> "$LOG" 2>&1
