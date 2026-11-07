#!/bin/bash
# Inspect the projection JSON structure before summarising it.
set -uo pipefail

source /local/ovi-env/bin/activate

python3 - <<'PY'
import json
from pathlib import Path

for name in ("projection_audio.json", "projection.json"):
    path = Path("/local/captures") / name
    if not path.exists():
        print(f"{name}: not present")
        continue
    data = json.loads(path.read_text())
    print(f"=== {name} ===")
    print("top-level type:", type(data).__name__)
    if isinstance(data, dict):
        print("keys:", list(data.keys()))
        for k, v in data.items():
            print(f"  {k}: {type(v).__name__}", end="")
            if isinstance(v, list):
                print(f" len={len(v)}")
                if v and isinstance(v[0], dict):
                    print("    first row keys:", list(v[0].keys()))
                    print("    first row:", json.dumps(v[0], default=str)[:400])
            else:
                print()
    else:
        print("len:", len(data))
        if data and isinstance(data[0], dict):
            print("first row keys:", list(data[0].keys()))
            print("first row:", json.dumps(data[0], default=str)[:400])
    print()
PY
