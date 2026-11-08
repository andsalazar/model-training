"""Capture Ovi residual states during a real generation and derive WSP directions.

What this produces
------------------
For each (stream, layer, step) it requests, a ``[n_patches, hidden]`` patch matrix
taken from the residual stream, plus a unit direction between two rollouts. The
direction is the instrument the project needs: a vector saying "the stable rollout's
state differs from the drifting rollout's state in this way, at this layer, at this
step".

Why two rollouts, not one
-------------------------
A direction is a contrast. The controlled pair here is a *drifting* run against a
*stable* run at the same prompt and seed: the earlier sweeps found 20 steps drifting
colour in 2 of 3 seeds and 30 steps stable. Subtracting the stable state from the
drifting one gives the edit an adapter would need to learn.

The two runs use different step counts, so a step index means a different timestep in
each. The report records the timestep for every step so the pair can be aligned by
noise level rather than by index -- comparing step 3 of 20 against step 3 of 30 would
compare two different points on the trajectory.

Cost
----
Full patch matrices are the expensive part. A 5s run at 704x704 is roughly 2400
patches and the video hidden width is 3072, so one (layer, step, stream) tensor is
~28MB in float32. Four layers x five steps x two streams is ~1.1GB, which fits, but
it is why the schedule is explicit rather than "capture everything".
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OVI_ROOT = Path(os.environ.get("OVI_ROOT", "/local/Ovi"))
if str(OVI_ROOT) not in sys.path:
    sys.path.insert(0, str(OVI_ROOT))

DEFAULT_LAYERS = (0, 1, 2, 4, 8)
DEFAULT_STEPS = (0, 2, 4, 6, 8)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Ovi DiT states.")
    parser.add_argument("--config", required=True, help="Ovi YAML config path")
    parser.add_argument("--prompt", required=True, help="text prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rollouts",
        type=int,
        nargs="+",
        default=[20, 30],
        help="one or more sample_steps values; each is a separate rollout",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument(
        "--capture-steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_STEPS),
        help="denoising step indices to record (0 is noisiest)",
    )
    parser.add_argument(
        "--capture-steps-for",
        nargs="+",
        default=[],
        metavar="STEPS=I,J,K",
        help=(
            "per-rollout capture steps, e.g. 30=0,3,6 (overrides --capture-steps for "
            "that rollout). Under UniPC the 20-step even indices and the 30-step "
            "every-third indices sit on identical timesteps."
        ),
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=["video", "audio"],
        choices=["video", "audio"],
        help="which of Ovi's two stacks to capture",
    )
    parser.add_argument(
        "--cfg-pass",
        default="cond",
        choices=["cond", "uncond"],
        help=(
            "which CFG pass to record. Ovi runs the conditional pass first and the "
            "unconditional (SLG) pass second at each timestep; before this option the "
            "second silently overwrote the first"
        ),
    )
    parser.add_argument("--out", required=True, type=Path, help="output .npz path")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help=(
            "override both negative prompts. By default the config's "
            "video_/audio_negative_prompt are used, as Ovi's inference.py does"
        ),
    )
    parser.add_argument(
        "--height-width",
        default=None,
        help=(
            "frame size as HxW, e.g. 704x704. Required in t2v/i2v modes: Ovi asserts "
            "it is not None before generating."
        ),
    )
    parser.add_argument(
        "--patch-stride",
        type=int,
        default=1,
        help=(
            "keep every Nth patch within each frame. A full 960x960 video residual is "
            "27900 tokens x 3072 wide = 171MB per (layer, step, stream) in float16, "
            "so a multi-layer pair of rollouts needs this."
        ),
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="frames in the rollout, so the stride stays frame-aligned",
    )
    return parser.parse_args(argv)


def _parse_height_width(text: str | None) -> list[int] | None:
    """Parse ``HxW`` into the ``[height, width]`` list Ovi expects."""
    if not text:
        return None
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"height-width must be HxW, got {text!r}")
    height, width = (int(p.strip()) for p in parts)
    if height <= 0 or width <= 0:
        raise ValueError(f"height-width must be positive, got {text!r}")
    return [height, width]


def _encode_video(video: np.ndarray) -> np.ndarray:
    """Store the decoded video compactly.

    The decoded output dominates the file: a 121-frame 960x960 rollout is 669MB as
    float16, against 21MB for one captured layer. It is kept for the colour and
    sharpness checks, but as uint8 — the quality metrics work on 8-bit frames anyway,
    and float16 decoded video buys nothing they can use.
    """
    array = np.asarray(video, dtype=np.float32)
    low, high = float(array.min()), float(array.max())
    span = high - low
    if span <= 0:
        return np.zeros(array.shape, dtype=np.uint8)
    # Ovi's decode output is in [-1, 1]; map that (and any other range) onto 0..255.
    scaled = (array - low) / span
    return np.rint(scaled * 255.0).astype(np.uint8)


def capture_steps_for(args: argparse.Namespace, steps: int) -> tuple[int, ...]:
    """Step indices to record for a rollout of ``steps`` steps."""
    for item in args.capture_steps_for:
        key, _, values = item.partition("=")
        if not values:
            raise ValueError(f"--capture-steps-for entry must be STEPS=I,J,K: {item!r}")
        if int(key) == steps:
            return tuple(int(v) for v in values.split(","))
    return tuple(args.capture_steps)


def run_rollout(
    engine, args, steps: int, capture, sampler: dict
) -> tuple[dict, object, object]:
    """One generation with hooks attached; returns the hook report and outputs."""
    from wsp.hooks import DitHooks

    hooks = DitHooks(
        engine.model, capture, cfg_pass=0 if args.cfg_pass == "cond" else 1
    )
    started = time.time()
    video = audio = None
    try:
        with hooks:
            video, audio, _image = engine.generate(
                text_prompt=args.prompt,
                seed=args.seed,
                sample_steps=steps,
                video_frame_height_width=_parse_height_width(args.height_width),
                **sampler,
            )
    finally:
        hooks.detach()
    report = hooks.report()
    report["seconds"] = round(time.time() - started, 1)
    report["sample_steps"] = steps
    report["seed"] = args.seed
    report["sampler"] = dict(sampler)
    # `None in (video, audio)` is wrong here: video and audio are numpy arrays, and
    # `in` against an array raises "truth value of an array is ambiguous". Compare
    # identities instead.
    if video is None or audio is None:
        report["notes"].append(
            "generate() returned None for at least one output: Ovi logs its own "
            "traceback and returns early rather than raising, so this rollout "
            "produced no output"
        )
    return report, video, audio


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from omegaconf import OmegaConf
    from ovi.ovi_fusion_engine import OviFusionEngine

    from wsp.dit import CaptureSchedule, ResidualCapture, ovi_sampler_kwargs

    config = OmegaConf.load(args.config)
    if args.max_frames is not None:
        config.max_frames = args.max_frames

    sampler = ovi_sampler_kwargs(config, args.negative_prompt)
    print(f"sampler: {sampler}", flush=True)

    print("loading engine...", flush=True)
    engine = OviFusionEngine(config=config, device=0)
    print("engine loaded", flush=True)

    schedule = CaptureSchedule(
        layers=tuple(args.layers),
        timesteps=tuple(args.capture_steps),
        keep_patches=True,
        streams=tuple(args.streams),
        patch_stride=args.patch_stride,
        n_frames=args.n_frames,
    )
    print(f"schedule: {schedule.describe()}", flush=True)
    if args.patch_stride > 1 and args.n_frames is None:
        print(
            "WARNING: patch_stride > 1 without --n-frames gives a flat stride, so the "
            "recorded token count may no longer divide evenly by the frame count and "
            "frame windows will raise",
            flush=True,
        )

    reports: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}

    def flush() -> None:
        """Write everything captured so far.

        Called after each rollout, not only at the end: a capture run is a long GPU
        job, and a failure in the last rollout should not discard the earlier ones.

        The arrays are already float16, so deflating them again is a poor trade --
        a 1.05GB smoke capture spent about six minutes compressing on top of 93
        seconds of generation. ``savez`` (uncompressed) keeps the write cost
        proportional to the data instead of to its entropy.
        """
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arrays)
        out.with_suffix(".json").write_text(json.dumps(reports, indent=2))
        size = out.stat().st_size / 1e6 if out.exists() else 0.0
        capture_bytes = sum(
            a.nbytes for k, a in arrays.items() if k.startswith("capture_")
        )
        output_bytes = sum(
            a.nbytes for k, a in arrays.items() if not k.startswith("capture_")
        )
        print(
            f"  (flushed {len(arrays)} arrays, {size:.0f} MB -> {out}; "
            f"captures {capture_bytes / 1e6:.0f} MB, decoded outputs "
            f"{output_bytes / 1e6:.0f} MB)",
            flush=True,
        )

    failures: list[str] = []
    for steps in args.rollouts:
        rollout_schedule = dataclasses.replace(
            schedule, timesteps=capture_steps_for(args, steps)
        )
        capture = ResidualCapture(schedule=rollout_schedule)
        print(f"\n=== rollout: {steps} steps, seed {args.seed} ===", flush=True)
        print(f"schedule: {rollout_schedule.describe()}", flush=True)
        try:
            report, video, audio = run_rollout(engine, args, steps, capture, sampler)
        except Exception as exc:  # noqa: BLE001 - a partial run is still useful
            failures.append(f"{steps} steps: {type(exc).__name__}: {exc}")
            print(f"ROLLOUT FAILED: {type(exc).__name__}: {exc}", flush=True)
            flush()
            continue
        reports[str(steps)] = report
        print(json.dumps(report, indent=2), flush=True)
        if video is not None:
            arrays[f"video_{steps}"] = _encode_video(video)
        if audio is not None:
            arrays[f"audio_{steps}"] = np.asarray(audio, dtype=np.float32)
        for (stream, layer, step), cap in capture.captures.items():
            if cap.patches is None:
                continue
            arrays[f"capture_{steps}_{stream}_{layer}_{step}"] = cap.patches.astype(
                np.float16
            )
        flush()

    if failures:
        reports["_failures"] = failures
        print("\n=== failures ===")
        for line in failures:
            print(" ", line)
        # The last flush already wrote the arrays; rewrite the report so the failure
        # list is on disk too.
        Path(args.out).with_suffix(".json").write_text(json.dumps(reports, indent=2))
    out = Path(args.out)
    if not out.exists():
        print(f"no output was written to {out}")
        return 1
    print(f"\ndone: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
