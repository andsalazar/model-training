#!/bin/bash
# Set up Ovi on the A100 for WSP activation work â€” the version that actually works.
#
# This consolidates what four separate scripts discovered the hard way. Each step
# below replaced a failure:
#
#   * Install to /local (container disk), NOT /workspace. The network volume
#     throttles to ~1 MB/s; a previous install took 94 min on a volume versus
#     3.5 min on local disk. The volume also enforces a 640 MB quota, which is
#     what silently killed the first pair capture.
#   * Pin torchvision and torchaudio to the torch release. Left unpinned, pip
#     resolves torchaudio to a CUDA-13 build and importing it fails with
#     "libcudart.so.13: cannot open shared object file" â€” which reads as a
#     missing library rather than a version mismatch.
#   * Use a PREBUILT FlashAttention wheel. The source build compiles five
#     architectures because flash-attention's setup.py reads FLASH_ATTN_CUDA_ARCHS
#     and ignores TORCH_CUDA_ARCH_LIST; it OOM-killed twice and never finished.
#     The prebuilt wheel takes seconds.
#   * Install einops explicitly. Ovi's requirements.txt omits it even though
#     ovi/modules/vae.py imports it at module scope.
#
# Only the 5s checkpoint: activation capture needs a forward pass, not video.
set -uo pipefail

WORK=/local
REPO="$WORK/Ovi"
VENV="$WORK/ovi-env"
LOG=/root/setup_v2.log

step() { echo; echo "=== $* ==="; }

step "environment"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
df -h "$WORK" | tail -1
python3 --version

step "clone"
if [ -d "$REPO/.git" ]; then
  cd "$REPO" && git pull --ff-only || true
else
  git clone --depth 1 https://github.com/character-ai/Ovi.git "$REPO"
fi
cd "$REPO" || exit 1

step "venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

step "torch (pinned)"
pip install --no-cache-dir \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" 2>&1 | tail -3
python3 -c "import torch, torchaudio, torchvision; \
print('torch', torch.__version__, 'audio', torchaudio.__version__, \
'vision', torchvision.__version__)"

step "requirements"
pip install --no-cache-dir -r requirements.txt 2>&1 | tail -5

step "einops (undeclared by Ovi)"
pip install --no-cache-dir einops 2>&1 | tail -2

step "weights (5s only)"
python3 download_weights.py --models 960x960_5s 2>&1 | tail -8

step "flash_attn (prebuilt wheel)"
# Try the wheel first. Only fall back to a source build if every wheel fails.
WHEELS=(
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
FA_OK=0
for url in "${WHEELS[@]}"; do
  echo "--- trying $(basename "$url") ---"
  if pip install --no-cache-dir --no-deps "$url" 2>&1 | tail -2; then
    if python3 -c "
import torch
from flash_attn import flash_attn_varlen_func
q = torch.randn(4, 8, 64, dtype=torch.float16, device='cuda')
cu = torch.tensor([0, 4], dtype=torch.int32, device='cuda')
out = flash_attn_varlen_func(q, q, q, cu, cu, 4, 4)
torch.cuda.synchronize()
print('SMOKE OK', tuple(out.shape), out.dtype)
" 2>&1 | tail -3; then
      FA_OK=1
      break
    fi
  fi
done
if [ "$FA_OK" -ne 1 ]; then
  echo "ABORT: no prebuilt wheel worked; a source build would be needed"
  exit 1
fi

step "verify"
python3 - <<'PY'
import torch, flash_attn
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("flash_attn", flash_attn.__version__)
q = torch.randn(2, 64, 8, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn.flash_attn_func(q, q, q)
torch.cuda.synchronize()
print("kernel OK", tuple(out.shape))
PY

step "ovi import"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO" || exit 1
python3 -c "import ovi.ovi_fusion_engine; print('ovi import OK')" 2>&1 | tail -5

step "checkpoints"
ls -la "$REPO/ckpts/Ovi/" 2>/dev/null | head -5
du -sh "$REPO/ckpts" 2>/dev/null

step "setup complete"
