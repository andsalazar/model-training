#!/usr/bin/env bash
# Verify the pod is ready for Stage 0.
set -eu

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== Python / torch ==="
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())'

echo "=== Disk ==="
df -h /workspace | tail -1

echo "=== Ready ==="
