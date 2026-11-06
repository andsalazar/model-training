"""Stage 6, step 1: does differential entropy separate a degrading rollout from a healthy one?

This is the cheapest decisive experiment in the whole Ovi project. TBC's adapter is
judged by differential entropy across generated frames as a proxy for information
collapse. If that metric cannot tell a rollout that degrades from one that does not,
on the same prompt, then there is no behaviour for WSP to measure and no target to
train an adapter against. That is worth knowing for a few dollars rather than after a
long training run.

The lesson from Stages 1-5 is that every failure came from measuring the wrong thing
or measuring it badly. So this script measures first and trains nothing.

Two modes, deliberately separate
--------------------------------
``synthetic``  take saved frames and construct a degrading variant by converging each
               frame progressively toward frame 0. This tests the *metric*, not the
               model: it asks whether entropy falls when frames genuinely converge.
               Runs anywhere, no GPU. This is the primary check.

``real``       generate with Ovi's own CLI and measure the videos it produced. This
               tests the *model*: whether real rollouts collapse in a way the metric
               catches. Needs a GPU with the model installed.

The synthetic path runs first because it isolates the metric from the model. If the
metric cannot separate a synthetic collapse it certainly cannot separate a real one.
Results are reported separately so a synthetic pass is never mistaken for evidence
about the model.

An earlier version of the synthetic construction blended each frame toward its
immediate predecessor. That reaches a stationary variance rather than converging, so
per-frame entropy fell once and then flattened — no trend. The metric correctly refused
to call it a collapse. The failure was in the fixture, not the metric.

Run as a file, not a heredoc: torch and diffusers both re-import main.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

import numpy as np  # noqa: E402

from wsp.entropy import (  # noqa: E402
    collapse_trend,
    entropy_series,
    frame_delta_entropy,
    metric_separates,
)

#: Prompts chosen to span the failure surface: a static scene (most likely to
#: collapse into a still), a scene with sustained motion (least likely), and one
#: with a repeating element (the classic degenerate attractor).
PROMPTS = [
    "A still photograph of a mountain lake at dawn, no motion.",
    "A river flowing through a forest, camera slowly panning.",
    "A ceiling fan spinning in an empty room.",
    "A person walking along a beach, waves in the background.",
    "A candle flame flickering in a dark room.",
]


def _load_frames(path: Path) -> np.ndarray:
    """Load a saved frame tensor as ``(n_frames, n_patches, n_dims)``.

    The generation script saves latents or decoded frames as ``.npy``; this keeps
    the metric script independent of how they were produced.
    """
    arr = np.load(path)
    if arr.ndim == 2:
        # A single frame's patches: treat as a one-frame rollout.
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected 2-D or 3-D frame array, got shape {arr.shape}")
    return arr


def _converge_toward_first(frames: np.ndarray, strength: float) -> np.ndarray:
    """Construct a degrading rollout by converging every frame toward the first.

    The convergence is *progressive*: frame ``i`` is blended toward frame 0 by
    ``strength * i / (n-1)``. So ``strength`` is the blend applied to the *last*
    frame, and the ramp runs from 0 at frame 0 to ``strength`` at the end. Frame 0
    is never modified.

    ``strength`` 0 leaves the rollout untouched; 1 makes the final frame identical
    to the first. Intermediate values make later frames progressively more like
    frame 0, which is what a collapsing rollout actually looks like: the model stops
    producing new information and keeps re-emitting what it already had.

    The ramp matters. A *uniform* blend would lower entropy without producing a
    trend, and a trend is the signal being tested — the metric requires a negative
    slope with a good fit, so a flat-but-lowered series is correctly rejected.

    An earlier version blended each frame toward its immediate predecessor. That
    reaches a stationary variance instead of converging, so per-frame entropy fell
    once and then flattened — no trend, and the metric correctly refused to call it
    a collapse. The failure was in the fixture, not the metric.

    This is a *metric* test, not a model test. It asks whether entropy falls when
    frames genuinely converge, holding everything else fixed.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    out = frames.copy()
    n = len(out)
    if n < 2:
        return out
    for i in range(1, n):
        alpha = strength * i / (n - 1)
        out[i] = (1.0 - alpha) * frames[i] + alpha * frames[0]
    return out


def _flatten_frames(frames: np.ndarray) -> list[np.ndarray]:
    """``(n_frames, n_patches, n_dims)`` -> one ``(n_patches, n_dims)`` array per frame."""
    return [frames[i] for i in range(frames.shape[0])]


def _subsample(frames: np.ndarray, max_patches: int) -> np.ndarray:
    """Cap patches per frame so the k-NN estimator stays tractable.

    The k-NN estimator is O(n^2) in patches. A 960x960 frame at patch size 16 is
    3600 patches, and 12 frames of that is 43M distance computations per frame —
    workable but slow. Subsampling to a few hundred patches per frame keeps the
    comparison valid because every frame is subsampled the same way.
    """
    if frames.shape[1] <= max_patches:
        return frames
    # Evenly spaced rather than random: deterministic, so a re-run reproduces.
    idx = np.linspace(0, frames.shape[1] - 1, max_patches).astype(int)
    return frames[:, idx, :]


def run_synthetic(args: argparse.Namespace) -> dict:
    """Test the metric against constructed collapses. No model needed."""
    frames = _subsample(_load_frames(Path(args.frames)), args.max_patches)
    flat = _flatten_frames(frames)
    print(f"loaded {frames.shape[0]} frames x {frames.shape[1]} patches x {frames.shape[2]} dims")

    healthy = entropy_series(flat, estimator=args.estimator)
    healthy_delta = frame_delta_entropy(flat, estimator=args.estimator)
    print(
        f"healthy   per-frame ratio={healthy.ratio:.4f} slope={healthy.slope:.4f} "
        f"r2={healthy.r_squared:.3f} collapsing={healthy.collapsing}"
    )
    print(
        f"healthy   delta     ratio={healthy_delta.ratio:.4f} slope={healthy_delta.slope:.4f} "
        f"r2={healthy_delta.r_squared:.3f} collapsing={healthy_delta.collapsing}"
    )

    results = []
    for strength in args.strengths:
        degraded = _converge_toward_first(frames, strength)
        flat_d = _flatten_frames(degraded)
        per_frame = entropy_series(flat_d, estimator=args.estimator)
        delta = frame_delta_entropy(flat_d, estimator=args.estimator)
        verdict = metric_separates(healthy, per_frame, min_gap=args.min_gap)
        delta_verdict = metric_separates(healthy_delta, delta, min_gap=args.min_gap)
        row = {
            "strength": strength,
            "per_frame": {
                "ratio": per_frame.ratio,
                "slope": per_frame.slope,
                "r_squared": per_frame.r_squared,
                "collapsing": per_frame.collapsing,
                "separates": verdict.separates,
                "gap": verdict.gap,
                "reason": verdict.reason,
            },
            "delta": {
                "ratio": delta.ratio,
                "slope": delta.slope,
                "r_squared": delta.r_squared,
                "collapsing": delta.collapsing,
                "separates": delta_verdict.separates,
                "gap": delta_verdict.gap,
                "reason": delta_verdict.reason,
            },
        }
        results.append(row)
        print(
            f"strength={strength:.2f}  per-frame: ratio={per_frame.ratio:.4f} "
            f"sep={verdict.separates} gap={verdict.gap:+.4f} | "
            f"delta: ratio={delta.ratio:.4f} sep={delta_verdict.separates} "
            f"gap={delta_verdict.gap:+.4f}"
        )

    # The metric is usable if it separates at least one constructed collapse on
    # either metric, and does not call the untouched rollout a collapse.
    per_frame_ok = any(r["per_frame"]["separates"] for r in results)
    delta_ok = any(r["delta"]["separates"] for r in results)
    return {
        "mode": "synthetic",
        "estimator": args.estimator,
        "frames": int(frames.shape[0]),
        "patches_per_frame": int(frames.shape[1]),
        "dims": int(frames.shape[2]),
        "healthy": healthy.as_dict(),
        "healthy_delta": healthy_delta.as_dict(),
        "results": results,
        "per_frame_separates": per_frame_ok,
        "delta_separates": delta_ok,
        "usable": bool((per_frame_ok or delta_ok) and not healthy.collapsing),
    }


def _extract_frames(video_path: Path, max_frames: int | None = None) -> np.ndarray:
    """Decode an mp4 into ``(n_frames, h, w, 3)`` uint8.

    Uses imageio-ffmpeg's bundled binary so no system ffmpeg is required. Frames
    are returned as-is; the metric flattens spatial patches itself.
    """
    import imageio.v3 as iio

    frames = iio.imread(video_path, plugin="pyav")
    if max_frames is not None:
        frames = frames[:max_frames]
    return np.asarray(frames)


def _patches_from_pixels(
    frames: np.ndarray, patch: int = 16, max_patches: int = 400
) -> np.ndarray:
    """Cut decoded frames into non-overlapping spatial patches.

    ``(n_frames, h, w, 3)`` -> ``(n_frames, n_patches, patch*patch*3)``. This is the
    pixel-space analogue of TBC's latent patches: same idea (a frame is a set of
    local tokens), different representation. The metric does not care which, as long
    as every frame is cut the same way.

    Patches are subsampled evenly to ``max_patches`` so the k-NN estimator stays
    tractable — a 960x960 frame at patch 16 is 3600 patches, and the estimator is
    O(n^2).
    """
    n_frames, h, w, channels = frames.shape
    ph, pw = h // patch, w // patch
    trimmed = frames[:, : ph * patch, : pw * patch, :]
    # (n, ph, patch, pw, patch, c) -> (n, ph*pw, patch*patch*c)
    grid = trimmed.reshape(n_frames, ph, patch, pw, patch, channels)
    grid = grid.transpose(0, 1, 3, 2, 4, 5)
    flat = grid.reshape(n_frames, ph * pw, patch * patch * channels)
    if flat.shape[1] > max_patches:
        idx = np.linspace(0, flat.shape[1] - 1, max_patches).astype(int)
        flat = flat[:, idx, :]
    return flat.astype(np.float64)


def run_real(args: argparse.Namespace) -> dict:
    """Generate real rollouts with Ovi's own CLI, then measure them.

    Ovi is not a diffusers pipeline — it is a custom repo driven by a YAML config
    and ``inference.py``. Rather than reimplement its inference here (which would
    couple this script to Ovi's internals and break on any upstream change), this
    shells out to Ovi's own entry point and then measures the videos it produced.

    That keeps the metric independent of the model, which is the whole point of
    building it as pure math first.

    Kept separate from the synthetic path so a synthetic pass is never reported as
    evidence about the model.
    """
    import subprocess

    repo = Path(args.ovi_repo)
    if not (repo / "inference.py").exists():
        raise SystemExit(
            f"Ovi repo not found at {repo}. Clone it first:\n"
            f"  git clone https://github.com/character-ai/Ovi.git {repo}"
        )
    config = repo / "ovi/configs/inference/inference_fusion.yaml"
    if not config.exists():
        raise SystemExit(f"Ovi config not found at {config}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ovi reads its prompt and output path from the YAML. Rather than editing the
    # repo's config in place (which would dirty a git checkout), write a copy with
    # the fields we need overridden.
    import yaml

    base = yaml.safe_load(config.read_text())
    base["output_dir"] = str(out_dir)
    base["ckpt_dir"] = args.ckpt_dir
    base["model_name"] = args.model_name
    base["sample_steps"] = args.steps
    base["cpu_offload"] = bool(args.offload)
    base["fp8"] = bool(args.fp8)
    base["seed"] = args.seed
    base["each_example_n_times"] = 1

    rows = []
    for i, prompt in enumerate(PROMPTS[: args.n_prompts]):
        print(f"[{i + 1}/{args.n_prompts}] {prompt}")
        run_config = dict(base)
        run_config["text_prompt"] = prompt
        run_config["output_dir"] = str(out_dir / f"run_{i:02d}")
        cfg_path = out_dir / f"config_{i:02d}.yaml"
        cfg_path.write_text(yaml.safe_dump(run_config))

        proc = subprocess.run(
            [sys.executable, "inference.py", "--config-file", str(cfg_path)],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"  generation failed (exit {proc.returncode})")
            print(proc.stderr[-2000:])
            rows.append({"prompt": prompt, "error": proc.stderr[-500:]})
            continue

        videos = sorted((out_dir / f"run_{i:02d}").rglob("*.mp4"))
        if not videos:
            print("  no mp4 produced")
            rows.append({"prompt": prompt, "error": "no mp4 produced"})
            continue

        frames = _extract_frames(videos[0])
        patches = _patches_from_pixels(frames, patch=args.patch, max_patches=args.max_patches)
        np.save(out_dir / f"patches_{i:02d}.npy", patches)
        print(f"  {frames.shape[0]} frames -> {patches.shape[1]} patches x {patches.shape[2]} dims")
        rows.append(
            {
                "prompt": prompt,
                "video": str(videos[0]),
                "frames": int(frames.shape[0]),
                "patches": int(patches.shape[1]),
                "dims": int(patches.shape[2]),
            }
        )

    return {"mode": "real", "model": args.model_name, "generations": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("synthetic", "real"),
        default="synthetic",
        help="synthetic tests the metric on saved frames; real generates with Ovi",
    )
    parser.add_argument("--frames", default="/workspace/ovi/frames.npy", help="saved frame tensor")
    parser.add_argument("--out", default="/workspace/ovi/metric.json", help="result JSON path")
    parser.add_argument("--out-dir", default="/workspace/ovi", help="generation output directory")
    parser.add_argument("--estimator", default="knn", choices=("histogram", "knn", "gaussian"))
    parser.add_argument("--max-patches", type=int, default=400)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--min-gap", type=float, default=0.15)
    parser.add_argument("--n-prompts", type=int, default=5)
    # -- real-generation options, matching Ovi's own interface ------------------
    parser.add_argument("--ovi-repo", default="/workspace/Ovi", help="path to the cloned Ovi repo")
    parser.add_argument("--ckpt-dir", default="/workspace/Ovi/ckpts", help="Ovi checkpoint directory")
    parser.add_argument(
        "--model-name",
        default="960x960_5s",
        choices=("720x720_5s", "960x960_5s", "960x960_10s"),
        help="Ovi model variant; 5s is enough to test the metric and is 2x cheaper",
    )
    parser.add_argument("--steps", type=int, default=50, help="denoising steps")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--patch", type=int, default=16, help="spatial patch size for pixel patches")
    parser.add_argument("--offload", action="store_true", help="CPU offload to cut peak VRAM")
    parser.add_argument("--fp8", action="store_true", help="load the fp8 checkpoint (24GB VRAM)")
    args = parser.parse_args()

    payload = run_synthetic(args) if args.mode == "synthetic" else run_real(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")

    if args.mode == "synthetic":
        print()
        print(f"per-frame metric separates: {payload['per_frame_separates']}")
        print(f"delta metric separates:     {payload['delta_separates']}")
        print(f"usable:                     {payload['usable']}")
        if not payload["usable"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
