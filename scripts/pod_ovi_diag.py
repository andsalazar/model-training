"""Diagnose the entropy estimator on real pixel patches.

The first real run gave a negative per-frame entropy and a nan ratio. That is the
k-NN estimator in a bad regime: 400 samples in 768 dimensions. Distance concentration
makes the estimate unreliable when n is not much larger than d.

This script measures the same video under several regimes so the right one can be
chosen from evidence rather than guessed:

  - patch size 16 (768 dims) vs 32 (3072 dims) vs 8 (192 dims)
  - patch count 400 vs 1000 vs 2000
  - raw patches vs PCA-reduced patches
  - k-NN vs gaussian estimator

The criterion is not "which gives the biggest number" but "which gives a stable,
positive estimate that still separates a constructed collapse".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

import numpy as np  # noqa: E402

from wsp.entropy import (  # noqa: E402
    entropy_series,
    frame_delta_entropy,
    gaussian_entropy,
    knn_entropy,
    metric_separates,
)


def load_video(path: Path) -> np.ndarray:
    import imageio.v3 as iio

    try:
        return np.asarray(iio.imread(path, plugin="FFMPEG"))
    except Exception:
        return np.asarray(iio.imread(path, plugin="pyav"))


def to_patches(frames: np.ndarray, patch: int, max_patches: int) -> np.ndarray:
    n, h, w, c = frames.shape
    ph, pw = h // patch, w // patch
    trimmed = frames[:, : ph * patch, : pw * patch, :]
    grid = trimmed.reshape(n, ph, patch, pw, patch, c).transpose(0, 1, 3, 2, 4, 5)
    flat = grid.reshape(n, ph * pw, patch * patch * c)
    if flat.shape[1] > max_patches:
        idx = np.linspace(0, flat.shape[1] - 1, max_patches).astype(int)
        flat = flat[:, idx, :]
    return flat.astype(np.float64)


def pca_reduce(patches: np.ndarray, n_components: int) -> np.ndarray:
    """Project patches onto their top principal components.

    Fit on all frames together so every frame is projected into the same basis —
    fitting per frame would make the entropies incomparable.
    """
    n_frames, n_patches, dim = patches.shape
    flat = patches.reshape(-1, dim)
    mean = flat.mean(axis=0)
    centred = flat - mean
    # Economy SVD; only the top components are needed.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    basis = vt[:n_components]
    reduced = centred @ basis.T
    return reduced.reshape(n_frames, n_patches, n_components)


def converge(patches: np.ndarray, strength: float) -> np.ndarray:
    out = patches.copy()
    n = len(out)
    for i in range(1, n):
        alpha = strength * i / (n - 1)
        out[i] = (1.0 - alpha) * patches[i] + alpha * patches[0]
    return out


def evaluate(patches: np.ndarray, label: str, estimator: str) -> dict:
    flat = [patches[i] for i in range(patches.shape[0])]
    pf = entropy_series(flat, estimator=estimator)
    dl = frame_delta_entropy(flat, estimator=estimator)

    # A usable regime gives a positive, finite per-frame entropy. Negative values
    # mean the estimator is out of its depth, not that the video has no information.
    usable = np.isfinite(pf.early_mean) and pf.early_mean > 0.0

    sep_pf = sep_dl = False
    if usable:
        for s in (0.5, 0.75, 0.9):
            deg = converge(patches, s)
            fd = [deg[i] for i in range(deg.shape[0])]
            if metric_separates(pf, entropy_series(fd, estimator=estimator)).separates:
                sep_pf = True
            if metric_separates(dl, frame_delta_entropy(fd, estimator=estimator)).separates:
                sep_dl = True

    print(
        f"{label:<34} n={patches.shape[1]:>5} d={patches.shape[2]:>5} "
        f"pf_mean={pf.early_mean:>9.2f} pf_ratio={pf.ratio:>7.3f} "
        f"dl_ratio={dl.ratio:>7.3f} usable={str(usable):<5} "
        f"sep_pf={str(sep_pf):<5} sep_dl={str(sep_dl):<5}"
    )
    return {
        "label": label,
        "n": int(patches.shape[1]),
        "d": int(patches.shape[2]),
        "pf_mean": float(pf.early_mean),
        "pf_ratio": float(pf.ratio),
        "dl_ratio": float(dl.ratio),
        "usable": bool(usable),
        "sep_pf": bool(sep_pf),
        "sep_dl": bool(sep_dl),
    }


def main() -> None:
    path = Path(sys.argv[1])
    frames = load_video(path)
    print(f"{path.name}: {frames.shape}\n")

    print("=== raw patches, varying patch size and count ===")
    for patch in (8, 16, 32):
        for count in (400, 1000, 2000):
            p = to_patches(frames, patch, count)
            evaluate(p, f"raw patch={patch} count={count}", "knn")

    print("\n=== PCA-reduced patches (patch=16, count=1000) ===")
    base = to_patches(frames, 16, 1000)
    for comps in (8, 16, 32, 64):
        p = pca_reduce(base, comps)
        evaluate(p, f"pca comps={comps}", "knn")

    print("\n=== gaussian estimator (patch=16, count=1000) ===")
    evaluate(base, "gaussian raw", "gaussian")
    for comps in (16, 32):
        p = pca_reduce(base, comps)
        evaluate(p, f"gaussian pca={comps}", "gaussian")

    print("\n=== direct estimator comparison on one frame ===")
    one = base[0]
    print(f"n={one.shape[0]} d={one.shape[1]}")
    print(f"  knn      {knn_entropy(one):.3f}")
    print(f"  gaussian {gaussian_entropy(one):.3f}")


if __name__ == "__main__":
    main()
