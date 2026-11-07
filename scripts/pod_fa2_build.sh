#!/bin/bash
# Rebuild FlashAttention 2 with the compiler error visible.
#
# The first attempt via `pip install` swallowed the error: the setup script piped the
# build through `tail -5`, so only "Error compiling objects for extension" survived and
# the actual compile failure was discarded. Capture the whole log this time.
set -uo pipefail

source /local/ovi-env/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4

cd /local/flash-attention || exit 1
echo "--- preconditions ---"
which nvcc gcc ninja
nvcc --version | tail -2
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "--- build (long) ---"
python3 setup.py bdist_wheel > /tmp/fa2build.log 2>&1
STATUS=$?
echo "build exit: $STATUS"
echo "--- error lines ---"
grep -n -E "error:|Error |fatal error|Killed|No such file" /tmp/fa2build.log | head -25
echo "--- last 30 lines ---"
tail -30 /tmp/fa2build.log
echo "--- dist ---"
ls -la dist/ 2>/dev/null || echo "no wheel produced"
