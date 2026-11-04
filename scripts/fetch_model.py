"""Download a checkpoint to a local directory on the pod."""

from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"
dest = sys.argv[2] if len(sys.argv) > 2 else "/workspace/qwen"

path = snapshot_download(
    repo, local_dir=dest, allow_patterns=["*.safetensors", "*.json", "*.txt"]
)
print("downloaded to", path)
