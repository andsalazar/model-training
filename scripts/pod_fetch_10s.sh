#!/bin/bash
# Fetch only the 10s Ovi checkpoint. Ovi's download_weights.py ignores --models (main()
# is called with output_dir only), so it would pull every variant.
set -uo pipefail
source /local/ovi-env/bin/activate
cd /local/Ovi
python3 - <<'PY'
import time
from huggingface_hub import snapshot_download
t = time.time()
snapshot_download(
    repo_id="chetwinlow1/Ovi",
    local_dir="./ckpts/Ovi",
    allow_patterns=["model_960x960_10s.safetensors"],
)
print(f"downloaded in {time.time() - t:.0f}s")
PY
ls -la /local/Ovi/ckpts/Ovi/
df -h /local | tail -1