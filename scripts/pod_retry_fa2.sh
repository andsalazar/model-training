#!/bin/bash
# Retry the FlashAttention build with the full compiler error visible.
#
# The setup script piped the build through `tail -5`, so the actual compiler error
# was discarded. Rebuild with output captured, and check the common causes first:
# nvcc presence, CUDA_HOME alignment with the installed torch, and compiler version.
set -uo pipefail

echo "--- nvcc ---"
which nvcc || echo "nvcc NOT on PATH"
/usr/local/cuda/bin/nvcc --version 2>&1 | tail -2
echo "--- torch cuda vs system cuda ---"
source /local/ovi-env/bin/activate
python3 -c "import torch; print('torch cuda', torch.version.cuda); print('cuda avail', torch.cuda.is_available())"
echo "--- gcc ---"
gcc --version | head -1
echo "--- disk ---"
df -h / | tail -1
echo "--- memory ---"
free -g | head -2
echo "--- cgroup mem limit ---"
cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null

cd /local/flash-attention || exit 1
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
echo "--- building (this is the long step) ---"
python3 setup.py bdist_wheel 2>&1 | grep -E "error|Error|ERROR|fatal|cannot|Killed|nvcc" | head -30
echo "--- exit: $? ---"
