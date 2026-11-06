#!/usr/bin/env bash
# Print the audio check report and confirm whether ffprobe exists at all.
#
# "streams=0" is ambiguous: it means either "the mp4 has no audio track" or "ffprobe is
# missing so nothing was inspected". Those are very different findings, so this prints
# the raw JSON plus the ffprobe path the script resolved.
set -u
cd /workspace
source /workspace/ovi-env/bin/activate

echo "=== ffprobe availability ==="
python3 - <<'PY'
from pathlib import Path
import imageio_ffmpeg
ff = Path(imageio_ffmpeg.get_ffmpeg_exe())
print("ffmpeg :", ff)
cand = ff.with_name(ff.name.replace("ffmpeg", "ffprobe"))
print("ffprobe:", cand, "exists" if cand.exists() else "MISSING")
PY

echo "=== report ==="
python3 - <<'PY'
import json
rows = json.load(open("/workspace/audio_check.json"))
for r in rows:
    print(r["dir"], "streams=", len(r.get("streams", [])), "audio=", r.get("audio_streams"))
    for s in r.get("streams", []):
        print("   ", s)
PY
