"""Run the collapse metric on real Ovi frames.

This is the decisive experiment. The synthetic tests established that the metric
detects a *constructed* collapse; this asks whether it detects a *real* one, and
whether a real healthy rollout is left alone.

The honest difficulty: a 5-second Ovi rollout may not collapse at all. If every real
generation is healthy, the metric has nothing to separate and the project needs a
different target. So this script reports the healthy readings first and only then
asks whether a constructed collapse on top of real frames is separable — which is a
weaker claim than "the metric catches real collapse", and is labelled as such.

Run on the pod inside the Ovi venv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

import numpy as np  # noqa: E402

from wsp.entropy import (  # noqa: E402
    entropy_series,
    frame_delta_entropy,
    metric_separates,
)


def load_video(path: Path) -> np.ndarray:
    import imageio.v3 as iio

    try:
        return np.asarray(iio.imread(path, plugin="FFMPEG"))
    except Exception:
        return np.asarray(iio.imread(path, plugin="pyav"))


def to_patches(frames: np.ndarray, patch: int = 16, max_patches: int = 400) -> np.ndarray:
    """``(n, h, w, 3)`` -> ``(n, n_patches, patch*patch*3)``, evenly subsampled."""
    n, h, w, c = frames.shape
    ph, pw = h // patch, w // patch
    trimmed = frames[:, : ph * patch, : pw * patch, :]
    grid = trimmed.reshape(n, ph, patch, pw, patch, c).transpose(0, 1, 3, 2, 4, 5)
    flat = grid.reshape(n, ph * pw, patch * patch * c)
    if flat.shape[1] > max_patches:
        idx = np.linspace(0, flat.shape[1] - 1, max_patches).astype(int)
        flat = flat[:, idx, :]
    return flat.astype(np.float64)


def converge(frames: np.ndarray, strength: float) -> np.ndarray:
    """Progressive convergence toward frame 0, matching pod_ovi_metric.py."""
    out = frames.copy()
    n = len(out)
    for i in range(1, n):
        alpha = strength * i / (n - 1)
        out[i] = (1.0 - alpha) * frames[i] + alpha * frames[0]
    return out


def pca_reduce(patches: np.ndarray, n_components: int) -> np.ndarray:
    """Project patches onto their top principal components.

    Raw pixel patches are 768-dimensional at patch size 16. Estimating a 768x768
    covariance from 1000 samples is a marginal regime — the matrix is near-singular
    and its log-determinant goes sharply negative, which is why the Gaussian
    estimator returned values like -1353 on real video while returning +956 on a
    different video. The number was measuring the conditioning of the covariance,
    not the information in the frame.

    PCA to a modest number of components fixes the conditioning. The basis is fit on
    all frames together so every frame is projected into the same space — fitting per
    frame would make the entropies incomparable.
    """
    n_frames, n_patches, dim = patches.shape
    if n_components >= dim:
        return patches
    flat = patches.reshape(-1, dim)
    mean = flat.mean(axis=0)
    centred = flat - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    basis = vt[:n_components]
    return (centred @ basis.T).reshape(n_frames, n_patches, n_components)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="mp4 paths")
    ap.add_argument("--out", default="/workspace/ovi_out/metric_real.json")
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--max-patches", type=int, default=400)
    ap.add_argument("--estimator", default="gaussian")
    ap.add_argument(
        "--pca",
        type=int,
        default=32,
        help="reduce patches to this many principal components; 0 disables",
    )
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.5, 0.75, 0.9])
    ap.add_argument("--min-gap", type=float, default=0.15)
    args = ap.parse_args()

    report = {"videos": [], "estimator": args.estimator}

    for path_str in args.videos:
        path = Path(path_str)
        print(f"\n=== {path.name} ===")
        frames = load_video(path)
        patches = to_patches(frames, args.patch, args.max_patches)
        if args.pca:
            patches = pca_reduce(patches, args.pca)
        print(f"frames {frames.shape[0]}  patches {patches.shape[1]}  dims {patches.shape[2]}")

        flat = [patches[i] for i in range(patches.shape[0])]
        per_frame = entropy_series(flat, estimator=args.estimator)
        delta = frame_delta_entropy(flat, estimator=args.estimator)
        print(
            f"REAL per-frame: ratio={per_frame.ratio:.4f} slope={per_frame.slope:.4f} "
            f"r2={per_frame.r_squared:.3f} collapsing={per_frame.collapsing}"
        )
        print(
            f"REAL delta    : ratio={delta.ratio:.4f} slope={delta.slope:.4f} "
            f"r2={delta.r_squared:.3f} collapsing={delta.collapsing}"
        )

        entry = {
            "video": path.name,
            "frames": int(frames.shape[0]),
            "patches": int(patches.shape[1]),
            "dims": int(patches.shape[2]),
            "real_per_frame": per_frame.as_dict(),
            "real_delta": delta.as_dict(),
            "constructed": [],
        }

        # Now ask the weaker question: on top of these real frames, does the metric
        # separate a constructed collapse? This is labelled "constructed" because it
        # is not evidence that real rollouts collapse.
        for strength in args.strengths:
            degraded = converge(patches, strength)
            flat_d = [degraded[i] for i in range(degraded.shape[0])]
            pf = entropy_series(flat_d, estimator=args.estimator)
            dl = frame_delta_entropy(flat_d, estimator=args.estimator)
            v_pf = metric_separates(per_frame, pf, min_gap=args.min_gap)
            v_dl = metric_separates(delta, dl, min_gap=args.min_gap)
            print(
                f"  constructed s={strength:.2f}  per-frame sep={v_pf.separates} "
                f"gap={v_pf.gap:+.4f} | delta sep={v_dl.separates} gap={v_dl.gap:+.4f}"
            )
            entry["constructed"].append(
                {
                    "strength": strength,
                    "per_frame": {
                        "ratio": pf.ratio,
                        "collapsing": pf.collapsing,
                        "separates": v_pf.separates,
                        "gap": v_pf.gap,
                        "reason": v_pf.reason,
                    },
                    "delta": {
                        "ratio": dl.ratio,
                        "collapsing": dl.collapsing,
                        "separates": v_dl.separates,
                        "gap": v_dl.gap,
                        "reason": v_dl.reason,
                    },
                }
            )

        report["videos"].append(entry)

    # Summary across videos.
    real_collapsed = sum(1 for v in report["videos"] if v["real_delta"]["collapsing"])
    pf_sep = sum(1 for v in report["videos"] if any(c["per_frame"]["separates"] for c in v["constructed"]))
    dl_sep = sum(1 for v in report["videos"] if any(c["delta"]["separates"] for c in v["constructed"]))
    report["summary"] = {
        "videos": len(report["videos"]),
        "real_rollouts_collapsing": real_collapsed,
        "videos_where_per_frame_separates_constructed": pf_sep,
        "videos_where_delta_separates_constructed": dl_sep,
    }
    print("\n=== summary ===")
    for k, v in report["summary"].items():
        print(f"{k}: {v}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
