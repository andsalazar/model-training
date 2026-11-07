#!/bin/bash
# Verify the flash_attn wheel against what Ovi actually calls.
#
# Ovi calls flash_attn.flash_attn_varlen_func with window_size=(-1,-1) and
# deterministic=False, on bf16 tensors. A wheel that passes a fp16 smoke test can still
# be wrong for the real call, so exercise the exact signature.
set -uo pipefail

source /local/ovi-env/bin/activate
export PYTHONPATH="/local/Ovi:${PYTHONPATH:-}"

echo "--- version ---"
python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)"

echo "--- Ovi's flag detection ---"
python3 -c "
from ovi.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
print('FA2 available:', FLASH_ATTN_2_AVAILABLE)
print('FA3 available:', FLASH_ATTN_3_AVAILABLE)
"

echo "--- backend name from wsp ---"
PYTHONPATH="/local/wsp/src:/local/Ovi:${PYTHONPATH:-}" python3 -c "
from wsp.hooks import attention_backend
print('attention_backend():', attention_backend())
"

echo "--- exact varlen signature (bf16, window_size, deterministic) ---"
python3 -c "
import torch
from flash_attn import flash_attn_varlen_func
b, lq, lk, h, d = 1, 128, 128, 8, 64
q = torch.randn(b * lq, h, d, dtype=torch.bfloat16, device='cuda')
k = torch.randn(b * lk, h, d, dtype=torch.bfloat16, device='cuda')
v = torch.randn(b * lk, h, d, dtype=torch.bfloat16, device='cuda')
cu = torch.tensor([0, lq], dtype=torch.int32, device='cuda')
out = flash_attn_varlen_func(
    q=q, k=k, v=v,
    cu_seqlens_q=cu, cu_seqlens_k=cu,
    max_seqlen_q=lq, max_seqlen_k=lk,
    dropout_p=0.0, softmax_scale=None, causal=False,
    window_size=(-1, -1), deterministic=False,
)
print('OVI-SIGNATURE OK', tuple(out.shape), out.dtype)
assert out.dtype == torch.bfloat16, out.dtype
assert not torch.isnan(out).any(), 'NaN in output'
print('no NaNs')
"
