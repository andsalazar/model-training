#!/usr/bin/env bash
# Set up Ovi on the A100 for WSP activation work.
#
# Differences from pod_ovi_setup_a100.sh:
#   * Installs to /local (container disk), NOT /workspace. The network volume
#     throttles to ~1 MB/s; a previous install took 94 min on a volume versus
#     3.5 min on local disk.
#   * Only the 5s checkpoint. Activation capture needs a forward pass, not video,
#     so the smaller model is enough and saves 23GB of download.
#
# Ovi hard-asserts FLASH_ATTN_2_AVAILABLE, so FlashAttention is required despite
# the README calling it optional. A100 is sm_80, which needs a source build on this
# torch/Python combination; that takes ~45 min.
set -euo pipefail

WORK=/local
REPO="$WORK/Ovi"
mkdir -p "$WORK"

echo "=== environment ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
df -h "$WORK" | tail -1
python3 --version

echo "=== clone ==="
if [ -d "$REPO/.git" ]; then
  cd "$REPO" && git pull --ff-only || true
else
  git clone --depth 1 https://github.com/character-ai/Ovi.git "$REPO"
fi
cd "$REPO"

echo "=== venv ==="
VENV="$WORK/ovi-env"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "=== torch ==="
# torchvision and torchaudio must be pinned to the torch release. Left unpinned,
# pip resolves torchaudio to a current release built against CUDA 13, and importing
# it fails with "libcudart.so.13: cannot open shared object file" because torch here
# is the cu124 build. Ovi imports torchaudio, so that failure is fatal, and it
# presents as a missing library rather than as a version mismatch.
pip install --no-cache-dir \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" 2>&1 | tail -3
python3 -c "import torch, torchaudio, torchvision; \
print('torch', torch.__version__, 'audio', torchaudio.__version__, \
'vision', torchvision.__version__)"

echo "=== requirements ==="
pip install --no-cache-dir -r requirements.txt 2>&1 | tail -5

echo "=== weights (5s only) ==="
python3 download_weights.py --models 960x960_5s 2>&1 | tail -8

echo "=== flash_attn (source build, sm_80) ==="
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8
pip install --no-cache-dir ninja 2>&1 | tail -2
cd "$WORK"
if [ ! -d flash-attention ]; then
  git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
fi
cd flash-attention
pip install --no-cache-dir --no-build-isolation . 2>&1 | tail -8

echo "=== verify ==="
python - <<'PY'
import torch, flash_attn
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("flash_attn", flash_attn.__version__)
q = torch.randn(2, 64, 8, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn.flash_attn_func(q, q, q)
torch.cuda.synchronize()
print("kernel OK", tuple(out.shape))
PY
ls -la "$REPO/ckpts/Ovi/" 2>/dev/null
echo "=== setup complete ==="
