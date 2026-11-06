#!/usr/bin/env bash
# Install a Blackwell-capable torch into the Ovi venv.
#
# The RTX PRO 4500 Blackwell reports compute capability sm_120. Torch 2.6.0+cu124
# supports only up to sm_90, so every CUDA kernel launch fails with "no kernel image
# is available for execution on the device". Ovi's requirements pin 2.6.0, but that
# pin predates Blackwell and cannot run on this GPU at all.
#
# The fix is a torch built against CUDA 12.8 or newer, which includes sm_120. Ovi's
# own code does not depend on 2.6.0-specific APIs, so a newer torch works; the pin is
# a reproducibility choice, not a hard requirement.
set -uo pipefail

source /workspace/ovi-env/bin/activate

echo "=== current torch ==="
python -c "import torch; print(torch.__version__); print('archs', torch.cuda.get_arch_list())" 2>&1 | grep -v Warning

echo "=== install torch with sm_120 support (cu128) ==="
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.1" torchvision torchaudio 2>&1 | tail -8

echo "=== verify ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("device capability", torch.cuda.get_device_capability())
print("arch list", torch.cuda.get_arch_list())
# Actually run a kernel: an arch list that includes sm_120 is not proof the build
# works. A real matmul is.
x = torch.randn(512, 512, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("matmul OK, sum", float(y.sum()))
PY
