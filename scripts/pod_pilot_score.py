"""Score the pilot's held-out videos (teacher + four variants) with one tool, on CPU."""
import json, sys
import numpy as np
import imageio.v3 as iio
sys.path.insert(0, "/local/wsp/src"); sys.path.insert(0, "/local/wsp/scripts")
from wsp.quality import measure_quality
from pod_ovi_colour_check import probe_frames

W = "/local/pilot5s"
out = {}
for v in ["teacher48", "base_unipc30", "base_unipc12", "base_euler12", "adapter_euler12"]:
    for s in (100, 200, 300):
        path = f"{W}/teacher/river_s{s}.mp4" if v == "teacher48" else f"{W}/eval/{v}_s{s}.mp4"
        frames = np.asarray(iio.imread(path, plugin="FFMPEG"))[..., :3]
        idx = np.linspace(0, len(frames) - 1, 48).astype(int)
        q = measure_quality(frames[idx]).as_dict()
        c = probe_frames(frames, v)
        out[f"{v}_s{s}"] = {**{k: round(x, 4) for k, x in q.items() if k != "n_frames"},
                            "colour_r2": c["mean_r2"], "colour_drift": c["mean_abs_drift"], "path": path}
json.dump(out, open(f"{W}/quality_all.json", "w"), indent=2)
print("wrote", len(out))