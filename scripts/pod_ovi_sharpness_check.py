"""Is rising sharpness at fewer steps real detail, or residual noise?

The 5s sweep found sharpness climbing monotonically as steps fall: 14.8 at 50 steps,
23.2 at 30, 28.7 at 20, 32.2 at 12. That direction is backwards from what a
diffusion model should do — fewer denoising steps means the last few steps of
noise removal never happen, so a partially denoised frame carries *noise*, and noise
has high Laplacian energy. So the metric may be rewarding exactly the artefact it
should penalise.

This distinguishes the two by looking at the *spatial structure* of the high-frequency
energy. Real detail is spatially organised (edges, texture with correlation). Residual
denoising noise is spatially random. Three probes:

1. **Noise floor test.** Measure sharpness on a temporally averaged frame vs each
   individual frame. Spatial detail survives averaging; independent noise cancels. If
   most of a video's sharpness disappears under averaging, it was noise.

2. **Autocorrelation test.** Detail has neighbours that correlate; noise does not.
   Report the lag-1 spatial autocorrelation of the high-pass residual.

3. **Frame-to-frame instability.** If a pixel's high-frequency content is noise, it
   differs between frames even when the underlying scene does not change. Report the
   temporal variance of the high-pass residual.

Run on the pod:
    python pod_ovi_sharpness_check.py /workspace/ovi_speed/*/...mp4 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(path: Path, max_frames: int = 48) -> np.ndarray:
    import imageio.v3 as iio

    frames = np.asarray(iio.imread(str(path), plugin="FFMPEG"))
    if frames.ndim == 4 and frames.shape[-1] == 4:
        frames = frames[..., :3]
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = frames[idx]
    return frames.astype(np.float64)


def luma(frames: np.ndarray) -> np.ndarray:
    if frames.shape[-1] >= 3:
        return 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
    return frames[..., 0]


def high_pass(frame: np.ndarray) -> np.ndarray:
    """Interior Laplacian, matching wsp.quality.sharpness."""
    centre = frame[1:-1, 1:-1]
    return (
        frame[:-2, 1:-1] + frame[2:, 1:-1] + frame[1:-1, :-2] + frame[1:-1, 2:]
        - 4.0 * centre
    )


def probe(path: Path, max_frames: int = 48) -> dict:
    frames = load(path, max_frames)
    y = luma(frames)

    hp = np.stack([high_pass(f) for f in y])
    raw_energy = float(np.mean(np.abs(hp)))

    # 1. Noise floor: average all frames, then measure. Coherent detail survives
    #    averaging; independent per-frame noise falls by roughly sqrt(n).
    averaged = y.mean(axis=0)
    avg_energy = float(np.mean(np.abs(high_pass(averaged))))

    # 2. Spatial autocorrelation of the high-pass residual at lag 1. Noise decorrelates.
    flat = hp.reshape(len(hp), -1)
    a = flat[:, :-1] - flat[:, :-1].mean(axis=1, keepdims=True)
    b = flat[:, 1:] - flat[:, 1:].mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ac = np.where(den > 1e-12, num / den, np.nan)
    spatial_ac = float(np.nanmean(ac))

    # 3. Temporal variance of the high-pass residual at each pixel, relative to its
    #    mean magnitude. Static scene + noisy residual gives a high value.
    temporal_var = float(np.mean(np.var(hp, axis=0)))
    relative = temporal_var / max(raw_energy**2, 1e-12)

    return {
        "file": path.name,
        "frames": int(len(frames)),
        "sharpness": round(raw_energy, 3),
        "averaged_sharpness": round(avg_energy, 3),
        # Below 1 means averaging reduced the high-frequency energy, i.e. part of it was
        # not consistent across frames.
        "survives_averaging": round(avg_energy / max(raw_energy, 1e-12), 3),
        "spatial_autocorrelation": round(spatial_ac, 4),
        "temporal_variance_ratio": round(relative, 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="/workspace/sharpness_check.json")
    ap.add_argument("--max-frames", type=int, default=48)
    args = ap.parse_args()

    rows = []
    for v in args.videos:
        path = Path(v)
        if not path.exists():
            print(f"MISSING {path}")
            continue
        row = probe(path, args.max_frames)
        rows.append(row)
        name = row["file"][:52]
        print(
            f"{name:<52} sharp={row['sharpness']:>7.2f} "
            f"surv={row['survives_averaging']:>5.2f} "
            f"spac_ac={row['spatial_autocorrelation']:>7.3f} "
            f"temp_var={row['temporal_variance_ratio']:>8.5f}"
        )

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
