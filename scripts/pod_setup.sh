#!/usr/bin/env bash
# Install Stage 0 dependencies on the pod.
#
# The image's system Python is PEP 668 managed, so packages go into a venv that
# inherits the image's torch rather than reinstalling it.
set -eu

VENV=/workspace/venv

if [ ! -d "$VENV" ]; then
  echo "=== creating venv (inherits system torch) ==="
  python3 -m venv --system-site-packages "$VENV"
fi

echo "=== installing python deps ==="
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet datasets accelerate safetensors

echo "=== versions ==="
"$VENV/bin/python" -c 'import torch, transformers, datasets; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); print("transformers", transformers.__version__); print("datasets", datasets.__version__)'

echo "=== done ==="