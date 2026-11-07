#!/bin/bash
# Run one short generation attached, with no hooks, to separate "generation fails" from
# "capture fails".
#
# Engine load is verified working (36.67 GB VRAM). The pair capture died with zero
# output, which a Python traceback cannot do -- so this tests whether the generation
# itself completes now that the process tree is not being detached.
set -uo pipefail

source /local/ovi-env/bin/activate
cd /local/Ovi || exit 1
export PYTHONPATH="/local/wsp/src:/local/Ovi:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== generation, no hooks, 4 steps, 704x704 ==="
python3 -u -c "
import logging, sys, time
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
from omegaconf import OmegaConf
from ovi.ovi_fusion_engine import OviFusionEngine
cfg = OmegaConf.load('/local/ovi/configs/inference_fusion_wsp.yaml')
engine = OviFusionEngine(config=cfg, device=0)
print('loaded', flush=True)
t0 = time.time()
out = engine.generate(
    text_prompt='A red fox in snow',
    seed=42,
    sample_steps=4,
    video_frame_height_width=[704, 704],
)
print('generate returned', type(out), flush=True)
if out is not None:
    vid, aud, img = out
    print('video', None if vid is None else vid.shape, flush=True)
    print('audio', None if aud is None else aud.shape, flush=True)
print('elapsed', round(time.time() - t0, 1), flush=True)
" 2>&1 | tail -30
echo "=== exit: $? ==="
