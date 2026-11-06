"""Stage 6, step 1: does differential entropy separate a degrading rollout from a healthy one?

This is the cheapest decisive experiment in the whole Ovi project. TBC's adapter is
judged by differential entropy across generated frames as a proxy for information
collapse. If that metric cannot tell a rollout that degrades from one that does not,
on the same prompt, then there is no behaviour for WSP to measure and no target to
train an adapter against. That is worth knowing for a few dollars rather than after a
long training run.

The lesson from Stages 1-5 is that every failure came from measuring the wrong thing
or measuring it badly. So this script measures first and trains nothing.

What "degrading" means here
---------------------------
Ovi generates 5 or 10 seconds. A 5-second rollout is short enough that collapse is
unlikely to appear on its own. So the script induces degradation deliberately, in two
ways, and asks whether the metric catches each:

  ``truncate``   generate at the model's normal settings, then keep only the first N
                 frames and re-run the *remaining* frames conditioned on a progressively
                 degraded context. This is the honest analogue of long-horizon rollout
                 failure, but it needs model access we may not have.

  ``synthetic``  take one real generation and construct a degrading variant by blending
                 each frame toward the previous one. This tests the *metric*, not the
                 model: it asks whether entropy falls when frames genuinely converge.
                 It is cheap, needs no extra generation, and is the primary check.

The synthetic path is the one that runs first, because it isolates the metric from the
model. If the metric cannot separate a synthetic collapse it certainly cannot separate a
real one. The real-generation path is the follow-up, and its results are reported
separately so a synthetic pass is never mistaken for a real one.

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

    ``strength`` 0 leaves the rollout untouched; 1 replaces every frame with the
    first. Intermediate values make later frames progressively more like frame 0,
    which is what a collapsing rollout actually looks like: the model stops
    producing new information and keeps re-emitting what it already had.

    The convergence is *progressive* — frame ``i`` is blended by ``strength * i /
    (n-1)`` — so the rollout degrades along its length rather than being uniformly
    blurred. A uniform blend would lower entropy without producing a trend, and a
    trend is the signal being tested.

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


def run_real(args: argparse.Namespace) -> dict:
    """Generate real rollouts with Ovi and measure them.

    Kept separate from the synthetic path so a synthetic pass is never reported as
    evidence about the model. Requires the model to be installed and loadable.
    """
    import torch
    from diffusers import DiffusionPipeline
    from diffusers.utils import export_to_video

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("no CUDA device; Ovi needs a GPU")

    print(f"loading {args.model} on {device}")
    pipe = DiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_model_cpu_offload") and args.offload:
        pipe.enable_model_cpu_offload()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, prompt in enumerate(PROMPTS[: args.n_prompts]):
        print(f"[{i + 1}/{args.n_prompts}] {prompt}")
        result = pipe(
            prompt=prompt,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
        )
        video = result.frames[0] if hasattr(result, "frames") else result[0]
        # Save the decoded frames so the metric can be recomputed without
        # regenerating — generation is the expensive part.
        arr = np.asarray(video, dtype=np.float32)
        np.save(out_dir / f"frames_{i:02d}.npy", arr)
        if args.export_mp4:
            export_to_video(video, str(out_dir / f"video_{i:02d}.mp4"), fps=args.fps)
        rows.append({"prompt": prompt, "shape": list(arr.shape)})

    return {"mode": "real", "model": args.model, "generations": rows}


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
    parser.add_argument("--model", default="chetwinlow1/Ovi")
    parser.add_argument("--estimator", default="knn", choices=("histogram", "knn", "gaussian"))
    parser.add_argument("--max-patches", type=int, default=400)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--min-gap", type=float, default=0.15)
    parser.add_argument("--n-prompts", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--export-mp4", action="store_true")
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
