#!/usr/bin/env bash
# Set up Ovi on an A100 80GB pod: clone, install deps, download BOTH checkpoints.
#
# Differences from pod_ovi_setup.sh (the Blackwell version):
#   * sm_80 has prebuilt FlashAttention wheels, so no source build and no SDPA shim.
#   * 80GB VRAM runs the 10s checkpoint in bf16 without fp8 or CPU offload.
#   * Both 5s and 10s checkpoints are downloaded, because the whole point is
#     comparing the short and long horizons.
#
# Everything goes on local container disk: a network volume throttles to ~1 MB/s and
# an earlier vLLM install took 94 minutes there versus 3.5 minutes on local disk.
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
# into it. A venv sidesteps that without --break-system-packages.
VENV="$WORK/ovi-env"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c "import sys; print('venv python', sys.version)"

echo "=== install torch ==="
# A100 is sm_80, which torch 2.6.0+cu124 supports natively. No Blackwell workaround.
pip install --no-cache-dir "torch==2.6.0" torchvision torchaudio 2>&1 | tail -5

echo "=== install requirements ==="
pip install --no-cache-dir -r requirements.txt 2>&1 | tail -10

echo "=== install flash_attn ==="
# Ovi hard-asserts FLASH_ATTN_2_AVAILABLE in its attention module, so this is not
# optional despite the repo's requirements listing it loosely. sm_80 has prebuilt
# wheels, so this should not need a source build.
pip install --no-cache-dir ninja 2>&1 | tail -2
pip install --no-cache-dir flash_attn 2>&1 | tail -5 || \
  echo "PREBUILT flash_attn FAILED - will need a source build"

echo "=== download weights (both horizons) ==="
python3 download_weights.py --models 960x960_5s 960x960_10s 2>&1 | tail -20

echo "=== verify ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("capability", torch.cuda.get_device_capability())
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
    q = torch.randn(2, 64, 8, 64, device="cuda", dtype=torch.bfloat16)
    out = flash_attn.flash_attn_func(q, q, q)
    torch.cuda.synchronize()
    print("flash_attn kernel OK", tuple(out.shape))
except Exception as exc:
    print("flash_attn FAILED:", type(exc).__name__, exc)
PY
ls -la "$REPO/ckpts/Ovi/" 2>/dev/null
echo "=== setup complete ==="
