#!/bin/bash
# Inspect a capture: was anything actually recorded, and are the states real?
#
# A directory listing showing a large npz is not evidence that a capture worked. This
# prints the hook report (fired counts, backend, recorded count) and then checks the
# arrays have structure: non-zero variance, plausible patch counts, finite values.
set -uo pipefail

source /local/ovi-env/bin/activate
export PYTHONPATH="/local/wsp/src:/local/Ovi:${PYTHONPATH:-}"

CAP="${1:-/local/captures/smoke.npz}"
echo "=== report json ==="
python3 - "$CAP" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
meta = path.with_suffix(".json")
if not meta.exists():
    print("NO REPORT JSON")
else:
    data = json.loads(meta.read_text())
    for key, report in data.items():
        if key.startswith("_"):
            print(f"{key}: {report}")
            continue
        print(f"--- rollout {key} steps, seed {report.get('seed')} ---")
        for field in (
            "attention_backend", "fired", "fired_total", "recorded",
            "decoded_declined", "steps_seen", "model_calls", "seconds", "notes",
        ):
            if field in report:
                print(f"  {field}: {report[field]}")
PY

echo
echo "=== array contents ==="
python3 - "$CAP" <<'PY'
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
with np.load(path) as data:
    print(f"keys: {len(data.files)}")
    captures = [k for k in data.files if k.startswith("capture_")]
    print(f"capture arrays: {len(captures)}")
    if not captures:
        print("NO CAPTURES -- the hooks recorded nothing")
        raise SystemExit(1)
    for key in sorted(captures)[:8]:
        array = data[key]
        finite = np.isfinite(array).all()
        std = float(np.asarray(array, dtype=np.float32).std())
        print(
            f"  {key}: shape={array.shape} dtype={array.dtype} "
            f"std={std:.4f} finite={finite}"
        )
    bad = [
        k for k in captures
        if not np.isfinite(data[k]).all() or float(np.asarray(data[k], np.float32).std()) == 0.0
    ]
    print(f"degenerate captures (non-finite or zero variance): {len(bad)}")
    for key in ("video_8", "audio_8"):
        if key in data.files:
            a = data[key]
            print(f"  {key}: shape={a.shape} dtype={a.dtype} finite={np.isfinite(a).all()}")
PY
