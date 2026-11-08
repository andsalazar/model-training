"""Colour verdict for the decoded videos stored in a capture-pair ``.npz``.

The drifting-versus-stable contrast assumes the 20-step rollout drifts and the 30-step
one does not. That was established by the sweeps over several prompts and seeds; this
checks it on the exact rollouts the direction was derived from.

The capture stores video as uint8 in Ovi's decode layout ``[C, F, H, W]``; the colour
probe wants ``[F, H, W, C]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pod_ovi_colour_check import probe_frames  # noqa: E402


def to_fhwc(video: np.ndarray) -> np.ndarray:
    """Reorder a decoded video to ``[F, H, W, C]``, whichever axis holds the channels."""
    array = np.asarray(video)
    if array.ndim != 4:
        raise ValueError(f"expected a 4-D video, got shape {array.shape}")
    # Ovi's own layout first: a tiny frame width could otherwise be read as channels.
    if array.shape[0] in (3, 4):
        return np.moveaxis(array, 0, -1)[..., :3]
    if array.shape[-1] in (3, 4):
        return array[..., :3]
    if array.shape[1] in (3, 4):
        return np.moveaxis(array, 1, -1)[..., :3]
    raise ValueError(f"no channel axis of size 3 or 4 in shape {array.shape}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    data = np.load(args.npz, mmap_mode="r")
    rows = []
    for key in sorted(k for k in data.files if k.startswith("video_")):
        frames = to_fhwc(data[key])
        row = probe_frames(frames, key)
        row["shape"] = list(frames.shape)
        rows.append(row)
        print(
            f"{key:<10} frames={row['frames']} cs={row['colour_stability']:>6.2f} "
            f"r2={row['mean_r2']:>5.3f} drift={row['mean_abs_drift']:>6.2f} "
            f"{row['verdict']}"
        )
    if not rows:
        print("no video_* arrays in the capture")
        return 1
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
