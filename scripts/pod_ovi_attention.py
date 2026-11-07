"""Capture Ovi attention maps during a real generation and contrast two rollouts.

What this produces
------------------
For each (layer, step) it requests, a reduced attention summary — per-head row entropy,
per-head column mass, and diagonal mass — taken from the real softmax over the real key
axis. Then a per-head contrast between the drifting (20-step) and stable (30-step)
rollouts, aligned by timestep.

Why this is the measurement the residual result motivates
---------------------------------------------------------
Stage 6c projected a residual-stream direction onto the two matrices that write to that
stream and found best alignment 0.129 against a 0.018 chance level. A residual state is a
*sum* of every component's contribution, so a direction derived from it is dominated by
whichever component contributes most — and there is no reason for that to be the one that
distinguishes drifting from stable. The attention pattern is not representable as a column
of ``o`` or ``ffn.2``: it lives in the ``[heads, query, key]`` tensor before the projection.

Cost
----
The full attention matrix is never materialised. A 960x960 5s rollout is 27,900 tokens, so
a full ``[heads, 27900, 27900]`` tensor would be 74 GB per layer. The summary is computed
in query chunks, so peak memory is ``chunk x keys`` per head. The capture itself is small:
a summary is a few hundred floats per (layer, step).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OVI_ROOT = Path(os.environ.get("OVI_ROOT", "/local/Ovi"))
if str(OVI_ROOT) not in sys.path:
    sys.path.insert(0, str(OVI_ROOT))

DEFAULT_LAYERS = (0, 2, 4, 8, 11)
DEFAULT_STEPS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Ovi attention maps.")
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
        "--chunk",
        type=int,
        default=512,
        help=(
            "query rows per chunk. Lower it if the card runs out of memory: peak "
            "memory is chunk x keys per head."
        ),
    )
    parser.add_argument("--out", required=True, type=Path, help="output .json path")
    parser.add_argument("--negative-prompt", default="", help="Ovi's negative prompt")
    parser.add_argument(
        "--height-width",
        default=None,
        help="frame size as HxW, e.g. 960x960. Required in t2v/i2v modes.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
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


def run_rollout(engine, args, steps: int, schedule) -> tuple[dict, object, object]:
    """One generation with attention hooks attached."""
    from wsp.attention import AttentionCapture
    from wsp.attention_hooks import AttentionHooks

    capture = AttentionCapture()
    hooks = AttentionHooks(
        model=engine.model, schedule=schedule, capture=capture, chunk=args.chunk
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
                video_negative_prompt=args.negative_prompt,
                audio_negative_prompt=args.negative_prompt,
            )
    finally:
        hooks.detach()
    report = hooks.report()
    report["seconds"] = round(time.time() - started, 1)
    report["sample_steps"] = steps
    report["seed"] = args.seed
    if video is None or audio is None:
        report["notes"].append(
            "generate() returned None for at least one output: Ovi logs its own "
            "traceback and returns early rather than raising"
        )
    return report, capture, video, audio


def _summaries(capture) -> dict:
    """Every captured summary as a JSON-safe dict, keyed by layer and step."""
    out = {}
    for (stream, layer, step), summary in capture.maps.items():
        out[f"{stream}_{layer}_{step}"] = summary.summary()
    return out


def _contrast(capture_a, capture_b, min_effect: float) -> list[dict]:
    """Per-head contrasts between two captures, aligned by timestep."""
    from wsp.attention import attention_contrast, head_ranking

    rows: list[dict] = []
    # Align by timestep, not by step index: a 20-step run and a 30-step run visit
    # different timesteps at the same index. The tracker's mapping is what makes the
    # comparison meaningful.
    common = set(capture_a.maps) & set(capture_b.maps)
    for key in sorted(common):
        stream, layer, step = key
        a = capture_a.maps[key]
        b = capture_b.maps[key]
        try:
            contrasts = attention_contrast(a, b, min_effect=min_effect)
        except ValueError as exc:
            rows.append({"key": f"{stream}_{layer}_{step}", "skipped": str(exc)})
            continue
        ranked = head_ranking(contrasts)
        rows.append(
            {
                "stream": stream,
                "layer": layer,
                "step": step,
                "n_heads_changed": len(contrasts),
                "top_heads": [c.summary() for c in ranked[:5]],
                "max_effect": ranked[0].effect_size if ranked else 0.0,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from omegaconf import OmegaConf
    from ovi.ovi_fusion_engine import OviFusionEngine

    from wsp.dit import CaptureSchedule

    config = OmegaConf.load(args.config)
    if args.max_frames is not None:
        config.max_frames = args.max_frames

    print("loading engine...", flush=True)
    engine = OviFusionEngine(config=config, device=0)
    print("engine loaded", flush=True)

    schedule = CaptureSchedule(
        layers=tuple(args.layers),
        timesteps=tuple(args.capture_steps),
        keep_patches=False,
        streams=("video",),
    )
    print(f"schedule: {schedule.describe()}", flush=True)

    reports: dict[str, dict] = {}
    captures: dict[int, object] = {}
    failures: list[str] = []

    for steps in args.rollouts:
        print(f"\n=== rollout: {steps} steps, seed {args.seed} ===", flush=True)
        try:
            report, capture, _video, _audio = run_rollout(engine, args, steps, schedule)
        except Exception as exc:  # noqa: BLE001 - a partial run is still useful
            failures.append(f"{steps} steps: {type(exc).__name__}: {exc}")
            print(f"ROLLOUT FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        reports[str(steps)] = report
        captures[steps] = capture
        print(json.dumps(report, indent=2), flush=True)

    payload: dict = {
        "prompt": args.prompt,
        "seed": args.seed,
        "layers": list(args.layers),
        "capture_steps": list(args.capture_steps),
        "chunk": args.chunk,
        "rollouts": reports,
        "summaries": {str(s): _summaries(c) for s, c in captures.items()},
    }

    # Contrast the first two rollouts, which is the drifting-versus-stable pair.
    if len(captures) >= 2:
        ordered = sorted(captures)
        a, b = ordered[0], ordered[1]
        payload["contrast"] = {
            "a_steps": a,
            "b_steps": b,
            "rows": _contrast(captures[a], captures[b], min_effect=0.05),
        }

    if failures:
        payload["_failures"] = failures
        print("\n=== failures ===")
        for line in failures:
            print(" ", line)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
