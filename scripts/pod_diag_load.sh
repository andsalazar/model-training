#!/bin/bash
# Isolate where the capture dies: does the engine even load now?
#
# The pair capture and the diagnostic both produced zero bytes of output and left no
# python process. The engine loaded fine earlier in the smoke run, so something changed
# since -- most likely disk (the container is at 74%) or the 37GB checkpoint on a
# container disk that also holds a 6.6GB venv.
set -uo pipefail

source /local/ovi-env/bin/activate
cd /local/Ovi || exit 1
export PYTHONPATH="/local/wsp/src:/local/Ovi:${PYTHONPATH:-}"

echo "=== disk before ==="
df -h / | tail -1
echo "=== import ==="
python3 -u -c "print('python alive', flush=True)"
python3 -u -c "
import torch
print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available(), flush=True)
print('free vram GB', torch.cuda.mem_get_info()[0] / 1e9, flush=True)
" 2>&1 | tail -5
echo "=== load engine ==="
python3 -u -c "
import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
from omegaconf import OmegaConf
from ovi.ovi_fusion_engine import OviFusionEngine
cfg = OmegaConf.load('/local/ovi/configs/inference_fusion_wsp.yaml')
print('config loaded', flush=True)
engine = OviFusionEngine(config=cfg, device=0)
print('ENGINE LOADED OK', flush=True)
import torch
print('vram after load GB', torch.cuda.memory_allocated() / 1e9, flush=True)
" 2>&1 | tail -25
echo "=== exit: $? ==="
df -h / | tail -1
