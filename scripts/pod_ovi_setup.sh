#!/usr/bin/env bash
# Set up Ovi on a fresh pod: clone the repo, install deps, download weights.
#
# Ovi is not a diffusers pipeline. It is a custom repo driven by a YAML config and
# inference.py, so setup means cloning it and running its own download script.
#
# The pod's container disk (/local or /) is ephemeral and much faster than a network
# volume, which throttles to ~1 MB/s. An earlier vLLM install took 94 minutes on a
# volume versus 3.5 minutes on local disk. So everything goes on local disk.
set -euo pipefail

WORK="${WORK:-/workspace}"
REPO="$WORK/Ovi"

echo "=== environment ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 --version
df -h "$WORK" | tail -2

echo "=== clone Ovi ==="
if [ -d "$REPO/.git" ]; then
  echo "repo already present, pulling"
  cd "$REPO" && git pull --ff-only || true
else
  git clone --depth 1 https://github.com/character-ai/Ovi.git "$REPO"
fi
cd "$REPO"

echo "=== create venv ==="
# The pod's system Python is externally managed (PEP 668), so pip refuses to install
# into it. A venv sidesteps that without --break-system-packages, which would risk
# breaking the image's own tooling.
VENV="$WORK/ovi-env"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c "import sys; print('venv python', sys.version)"

echo "=== install torch ==="
# Ovi pins torch 2.6.0. The pod image ships torch 2.8.0 in system Python; the venv
# starts clean, so 2.6.0 installs without fighting the image.
python -c "import torch; print('torch', torch.__version__)" 2>/dev/null || true
pip install --no-cache-dir "torch==2.6.0" torchvision torchaudio 2>&1 | tail -5

echo "=== install requirements ==="
pip install --no-cache-dir -r requirements.txt 2>&1 | tail -10

echo "=== install flash_attn (optional, may fail) ==="
# FlashAttention is optional: Ovi runs without it, just slower. A failure here must
# not abort setup. It also needs ninja on PATH to compile.
pip install --no-cache-dir ninja 2>&1 | tail -2
pip install --no-cache-dir flash_attn --no-build-isolation 2>&1 | tail -5 || \
  echo "flash_attn install failed - continuing without it"

echo "=== download weights ==="
# The 5s variant is enough to test the metric and is half the cost of 10s.
python3 download_weights.py --models 960x960_5s 2>&1 | tail -20

echo "=== verify ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
ls -la "$REPO/ckpts" 2>/dev/null | head -20
echo "=== setup complete ==="
