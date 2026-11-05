#!/bin/bash
# Report what a pod actually has: vLLM, torch, the model, and free space.
set -u
PY=/workspace/venv2/bin/python

echo "=== PYTHON ==="
$PY -V 2>&1

echo "=== PACKAGES ==="
$PY - <<'EOF' 2>&1 | tail -6
try:
    import vllm, torch, transformers
    print("vllm", vllm.__version__)
    print("torch", torch.__version__)
    print("transformers", transformers.__version__)
    print("cuda", torch.cuda.is_available())
except Exception as exc:
    print("IMPORT_FAIL", type(exc).__name__, exc)
EOF

echo "=== MODEL SEARCH ==="
for d in /workspace/qwen /workspace/models /workspace/model /root/qwen; do
  [ -d "$d" ] && echo "found dir: $d" && ls "$d" | head -5
done
find /workspace -maxdepth 3 -name "*.safetensors" 2>/dev/null | head -5
echo "config.json hits:"
find /workspace -maxdepth 3 -name "config.json" 2>/dev/null | head -5

echo "=== DISK ==="
df -h /workspace / 2>&1 | tail -3

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>&1
