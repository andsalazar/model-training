#!/usr/bin/env bash
# Check which video decoder backends are available in the Ovi venv.
#
# The first sweep died because imageio's pyav plugin was requested and not installed.
# This reports what IS available so the fallback chain in pod_ovi_speed.py can be
# verified against reality rather than assumed.
set -uo pipefail

source /workspace/ovi-env/bin/activate

echo "=== generated videos ==="
find /workspace/ovi_speed -name '*.mp4' 2>/dev/null | sort

echo "=== decoder backends ==="
python - <<'PY'
import importlib.util

for name in ("imageio", "imageio_ffmpeg", "av", "cv2"):
    spec = importlib.util.find_spec(name)
    print(f"  {name}: {'present' if spec else 'MISSING'}")

print("=== decode attempt ===")
import glob
import imageio.v3 as iio

videos = sorted(glob.glob("/workspace/ovi_speed/**/*.mp4", recursive=True))
if not videos:
    print("  no videos to decode")
else:
    video = videos[0]
    print(f"  trying {video}")
    for plugin in ("FFMPEG", "pyav", "ffmpeg", None):
        try:
            arr = iio.imread(video, plugin=plugin) if plugin else iio.imread(video)
            import numpy as np
            a = np.asarray(arr)
            print(f"  plugin={plugin}: OK shape={a.shape} dtype={a.dtype}")
            break
        except Exception as exc:
            print(f"  plugin={plugin}: {type(exc).__name__}: {str(exc)[:120]}")
PY
