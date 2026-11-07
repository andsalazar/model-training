#!/bin/bash
# Confirm Ovi imports cleanly from the repo root and report the attention backend.
# The previous attempt ran from /root, where Ovi's default relative config path does
# not resolve -- that is a working-directory issue, not a missing dependency.
set -uo pipefail

source /local/ovi-env/bin/activate
cd /local/Ovi || exit 1
export PYTHONPATH="/local/wsp/src:/local/Ovi:${PYTHONPATH:-}"
export OVI_ROOT=/local/Ovi

echo "=== import Ovi from repo root ==="
python3 -c "
import ovi.ovi_fusion_engine as e
print('OviFusionEngine importable:', hasattr(e, 'OviFusionEngine'))
" 2>&1 | tail -5

echo "=== attention backend via Ovi flags ==="
python3 -c "
from ovi.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
print('FA2:', FLASH_ATTN_2_AVAILABLE, 'FA3:', FLASH_ATTN_3_AVAILABLE)
" 2>&1 | tail -3

echo "=== attention_backend() from wsp ==="
python3 -c "
from wsp.hooks import attention_backend
b = attention_backend()
print('backend:', b)
raise SystemExit(0 if b == 'flash_attn_2' else 1)
" 2>&1 | tail -3
echo "backend check exit: $?"

echo "=== wsp modules importable ==="
python3 -c "
from wsp.dit import CaptureSchedule, ResidualCapture
from wsp.hooks import DitHooks, attention_backend
from wsp.targeting import build_plan, apply_plan
print('wsp import OK')
"

echo "=== checkpoints present ==="
ls /local/Ovi/ckpts/Ovi/ 2>/dev/null | head -5
du -sh /local/Ovi/ckpts 2>/dev/null
