#!/bin/bash
# Download Qwen3.5-0.8B onto the pod's local disk.
#
# The pod's /workspace network volume has no model and throttles badly, so the
# model goes to /local/models and vLLM loads it from there.
set -u
LOG=/workspace/fetch_model.log
: > "$LOG"

{
  echo "=== START $(date -u +%H:%M:%S) ==="
  mkdir -p /local/models
  export HF_HOME=/local/hf
  export HF_HUB_ENABLE_HF_TRANSFER=0

  /local/venv/bin/python - <<'EOF' 2>&1 | tail -20
from huggingface_hub import snapshot_download
path = snapshot_download(
    "Qwen/Qwen3.5-0.8B",
    local_dir="/local/models/qwen",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
)
print("MODEL_AT", path)
EOF
  echo "download_rc=$?"

  echo "=== FILES ==="
  ls -la /local/models/qwen 2>&1 | head -15
  du -sh /local/models/qwen 2>/dev/null

  echo "FETCH_DONE"
} >> "$LOG" 2>&1
