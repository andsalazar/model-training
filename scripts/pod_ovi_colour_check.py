"""Is the rising colour_stability at low step counts drift, or a measurement artefact?

At the 10s horizon colour_stability climbs with fewer steps: 2.83 at 50, 5.32 at 30,
6.77 at 20. That metric is the spread of per-frame MEAN colour across the rollout, so
a rise means frames differ from each other more in overall hue or exposure.

Two very different causes fit the number:

1. **Real drift.** The rollout starts in one colour balance and ends in another. That
   is a visible defect — the scene changes tone for no reason.
2. **More saturation/variety, not drift.** If frames oscillate around a stable mean
   rather than trending, the spread is larger without anything being wrong.

The distinguishing test is *trend vs oscillation*. This fits a line to per-frame mean
colour and reports:

  - the slope (drift) relative to the spread
  - the fraction of variance explained by the trend (R^2)
  - the mean of the first third vs the last third, per channel

A real drift shows a monotone slope with high R^2 and a large early/late gap.
Oscillation shows a near-zero slope with low R^2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(path: Path, max_frames: int = 121) -> np.ndarray:
    import imageio.v3 as iio

    frames = np.asarray(iio.imread(str(path), plugin="FFMPEG"))
    if frames.ndim == 4 and frames.shape[-1] == 4:
        frames = frames[..., :3]
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = frames[idx]
    return frames.astype(np.float64)


def _fit(y: np.ndarray) -> tuple[float, float]:
    """Slope and R^2 of y against its index."""
    n = len(y)
    if n < 2:
        return 0.0, 0.0
    x = np.arange(n, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom == 0.0:
        return 0.0, 0.0
    slope = float(((x - xm) * (y - ym)).sum() / denom)
    pred = ym + slope * (x - xm)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return slope, r2


def probe(path: Path, max_frames: int = 121) -> dict:
    return probe_frames(load(path, max_frames), path.name)


def probe_frames(frames: np.ndarray, name: str) -> dict:
    """The colour verdict for an ``[F, H, W, C]`` frame array."""
    frames = np.asarray(frames, dtype=np.float64)
    # Per-frame mean colour, one value per channel.
    per_frame = frames.mean(axis=(1, 2))

    channels = {}
    for idx, name in enumerate(("R", "G", "B")[: per_frame.shape[1]]):
        series = per_frame[:, idx]
        slope, r2 = _fit(series)
        third = max(1, len(series) // 3)
        early = float(series[:third].mean())
        late = float(series[-third:].mean())
        channels[name] = {
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "slope_per_frame": round(slope, 4),
            "r2": round(float(r2), 3),
            "early": round(early, 2),
            "late": round(late, 2),
            "drift": round(late - early, 2),
        }

    spread = float(np.mean(np.std(per_frame, axis=0)))
    # How much of the spread is explained by a monotone trend, averaged over channels.
    mean_r2 = float(np.mean([c["r2"] for c in channels.values()]))
    mean_abs_drift = float(np.mean([abs(c["drift"]) for c in channels.values()]))
    monotone = mean_r2 > 0.5

    return {
        "file": name,
        "frames": int(len(frames)),
        "colour_stability": round(spread, 3),
        "mean_r2": round(mean_r2, 3),
        "mean_abs_drift": round(mean_abs_drift, 2),
        "monotone_drift": monotone,
        "verdict": (
            "drift - colour trends monotonically across the rollout"
            if monotone
            else "oscillation - frames vary around a stable mean, no trend"
        ),
        "channels": channels,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="/workspace/colour_check.json")
    args = ap.parse_args()

    rows = []
    for v in args.videos:
        path = Path(v)
        if not path.exists():
            print(f"MISSING {path}")
            continue
        row = probe(path)
        rows.append(row)
        name = row["file"][:44]
        verdict = row["verdict"][:44]
        print(
            f"{name:<44} cs={row['colour_stability']:>6.2f} "
            f"r2={row['mean_r2']:>5.3f} drift={row['mean_abs_drift']:>6.2f} {verdict}"
        )

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
