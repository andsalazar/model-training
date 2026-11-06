"""Probe a generated Ovi video: shape, frame count, and whether it is not degenerate.

Run on the pod inside the Ovi venv. Prints facts only — no interpretation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> None:
    path = Path(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    import imageio.v3 as iio

    # pyav is not installed in the Ovi venv; the ffmpeg backend is, via
    # imageio-ffmpeg. Try ffmpeg first and fall back so this works either way.
    try:
        frames = iio.imread(path, plugin="FFMPEG")
    except Exception:
        frames = iio.imread(path, plugin="pyav")
    print(f"file      {path.name}")
    print(f"size      {path.stat().st_size / 1e6:.1f} MB")
    print(f"shape     {frames.shape}")
    print(f"dtype     {frames.dtype}")

    arr = np.asarray(frames, dtype=np.float32)
    print(f"range     [{arr.min():.1f}, {arr.max():.1f}]")
    print(f"mean      {arr.mean():.2f}")

    # Per-frame mean absolute difference from the previous frame. A degenerate
    # rollout (all frames identical) gives ~0; a healthy one gives a real number.
    if arr.shape[0] > 1:
        diffs = np.abs(np.diff(arr, axis=0)).mean(axis=(1, 2, 3))
        print(f"frame diff mean {diffs.mean():.3f} min {diffs.min():.3f} max {diffs.max():.3f}")
        # A monotone fall in frame-to-frame difference is the visual signature of
        # the collapse the metric is meant to catch.
        first_third = diffs[: max(1, len(diffs) // 3)].mean()
        last_third = diffs[-max(1, len(diffs) // 3) :].mean()
        print(f"diff early {first_third:.3f} late {last_third:.3f} ratio {last_third / first_third:.3f}")

    # Distinct-frame count: how many frames differ from their predecessor by more
    # than a small threshold. A collapsed rollout has very few.
    if arr.shape[0] > 1:
        per_frame = np.abs(np.diff(arr, axis=0)).mean(axis=(1, 2, 3))
        print(f"frames with diff > 1.0: {(per_frame > 1.0).sum()} of {len(per_frame)}")


if __name__ == "__main__":
    main()
