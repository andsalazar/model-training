import json, sys
import numpy as np, imageio.v3 as iio
out = {}
for name, path in [("teacher48", "/local/pilot5s/teacher/violin_s100.mp4"),
                   ("adapter_euler12", "/local/adapter5s_v2/eval/adapter_euler12_violin_s100.mp4"),
                   ("base_unipc12", "/local/adapter5s_v2/eval/base_unipc12_violin_s100.mp4"),
                   ("base_unipc30", "/local/adapter5s_v2/eval/base_unipc30_violin_s100.mp4")]:
    f = np.asarray(iio.imread(path, plugin="FFMPEG"))[..., :3].astype(np.float32)
    luma = (f[..., 0] * .299 + f[..., 1] * .587 + f[..., 2] * .114).mean(axis=(1, 2))
    jumps = np.abs(np.diff(luma))
    spec = np.abs(np.fft.rfft(luma - luma.mean()))
    period = len(luma) / max(int(np.argmax(spec[1:]) + 1), 1)
    big = np.where(jumps > 3 * np.median(jumps) + 0.5)[0]
    out[name] = {"frames": len(luma), "luma_mean": round(float(luma.mean()), 2), "luma_std": round(float(luma.std()), 2),
                 "max_jump": round(float(jumps.max()), 2), "median_jump": round(float(np.median(jumps)), 3),
                 "dominant_period_frames": round(period, 2), "big_jump_frames": big.tolist()[:30],
                 "luma_first_last_10": [round(float(luma[:10].mean()), 1), round(float(luma[-10:].mean()), 1)]}
    print(name, json.dumps(out[name]))