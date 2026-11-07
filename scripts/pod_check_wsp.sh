#!/bin/bash
# Confirm the wsp modules import on the pod and report the attention backend.
#
# Run as a script file rather than an inline ssh command: nested quoting through
# PowerShell -> ssh -> bash mangles anything containing quotes or parentheses.
set -uo pipefail

source /local/ovi-env/bin/activate
export PYTHONPATH="/local/wsp/src:/local/Ovi:${PYTHONPATH:-}"

echo "=== wsp import ==="
python3 -c "import wsp; print('wsp OK', wsp.__file__)" 2>&1 | tail -3

echo "=== attention backend ==="
python3 -c "from wsp.hooks import attention_backend; print('backend:', attention_backend())" 2>&1 | tail -3

echo "=== attention module ==="
python3 -c "
import numpy as np
from wsp.attention import summarize_attention, attention_contrast
a = np.full((2, 4, 4), 0.25, dtype=np.float32)
s = summarize_attention(a)
print('row_entropy', s.row_entropy)
print('diag', s.diagonal_mass)
print('summary', s.summary())
" 2>&1 | tail -6

echo "=== checkpoints ==="
ls /local/Ovi/ckpts/Ovi/ 2>/dev/null | head -3
