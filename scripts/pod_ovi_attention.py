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

DEFAULT_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27)
DEFAULT_STEPS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)
#: Per-rollout capture steps chosen so both runs sit on the *same* timesteps. Under
#: Ovi's UniPC schedule (shift 5.0) the 20-step run's even indices and the 30-step
#: run's every-third indices land on identical values: 999, 978, 952, 920, 882, 833,
#: 768, 681, 555, 356. Capturing those makes every contrast row an exact match.
DEFAULT_STEPS_FOR = {
    20: (0, 2, 4, 6, 8, 10, 12, 14, 16, 18),
    30: (0, 3, 6, 9, 12, 15, 18, 21, 24, 27),
}


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
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYERS),
        help="block indices to record, in each stream (Ovi has 30 per stream)",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=["video", "audio"],
        choices=["video", "audio"],
    )
    parser.add_argument(
        "--capture-steps",
        type=int,
        nargs="+",
        default=None,
        help=(
            "denoising step indices to record (0 is noisiest), for rollouts without a "
            "per-rollout entry. Defaults to timestep-matched steps for 20 and 30."
        ),
    )
    parser.add_argument(
        "--capture-steps-for",
        nargs="+",
        default=[],
        metavar="STEPS=I,J,K",
        help="per-rollout capture steps, e.g. 30=0,3,6,9 (overrides --capture-steps)",
    )
    parser.add_argument(
        "--cfg-pass",
        default="cond",
        choices=["cond", "uncond"],
        help=(
            "which CFG pass to record. Ovi runs the conditional pass first and the "
            "unconditional (SLG) pass second at each timestep."
        ),
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
        help="frame size as HxW, e.g. 960x960. Required in t2v/i2v modes.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--max-gap",
        type=float,
        default=1.0,
        help=(
            "largest timestep gap allowed when pairing captured steps across runs. "
            "With the default timestep-matched schedules every pair is exact, so a "
            "non-zero gap means the schedules were overridden; a loose tolerance would "
            "contrast different points on the trajectory and call it drift."
        ),
    )
    return parser.parse_args(argv)


def capture_steps_for(args: argparse.Namespace, steps: int) -> tuple[int, ...] | None:
    """Which step indices to record for a rollout of ``steps`` steps.

    Priority: ``--capture-steps-for``, then ``--capture-steps``, then the
    timestep-matched defaults, then ``None`` (every step).
    """
    for item in args.capture_steps_for:
        key, _, values = item.partition("=")
        if not values:
            raise ValueError(f"--capture-steps-for entry must be STEPS=I,J,K: {item!r}")
        if int(key) == steps:
            return tuple(int(v) for v in values.split(","))
    if args.capture_steps is not None:
        return tuple(args.capture_steps)
    return DEFAULT_STEPS_FOR.get(steps)


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


def run_rollout(
    engine, args, steps: int, schedule, sampler: dict
) -> tuple[dict, object, object, object]:
    """One generation with attention hooks attached."""
    from wsp.attention import AttentionCapture
    from wsp.attention_hooks import AttentionHooks

    capture = AttentionCapture()
    hooks = AttentionHooks(
        model=engine.model,
        schedule=schedule,
        capture=capture,
        chunk=args.chunk,
        cfg_pass=0 if args.cfg_pass == "cond" else 1,
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


def _contrast(
    capture_a,
    capture_b,
    timesteps_a: dict,
    timesteps_b: dict,
    min_effect: float,
    max_gap: float = 50.0,
) -> list[dict]:
    """Per-head contrasts between two captures, aligned by real timestep.

    Aligning by step index would be wrong. A 20-step run and a 30-step run visit
    different noise levels at the same index: under Ovi's UniPC schedule, index 18 is
    t=356 in the 20-step run and t=768 in the 30-step run. Comparing those compares two
    different points on the trajectory, so "the effect grows with step" would be an
    artefact of the schedule rather than a measurement of drift.

    Pairing is over *captured* steps only. An earlier version paired each step of run A
    with the nearest step of run B's full schedule and then silently skipped the row
    when that step had not been captured, so most requested timesteps vanished without
    a trace. Now each captured (stream, block, step) in A is paired with the nearest
    captured step for the same (stream, block) in B; a pair beyond ``max_gap`` is kept
    as a ``skipped`` row with the reason, never dropped.
    """
    from wsp.attention import attention_contrast, head_ranking

    time_a = {int(k): float(v) for k, v in timesteps_a.items()}
    time_b = {int(k): float(v) for k, v in timesteps_b.items()}
    if not time_a or not time_b:
        return []

    rows: list[dict] = []
    for stream, layer, step_a in sorted(capture_a.maps):
        t_a = time_a.get(step_a)
        if t_a is None:
            rows.append(
                {
                    "stream": stream,
                    "layer": layer,
                    "step_a": step_a,
                    "skipped": f"run A has no timestep recorded for step {step_a}",
                }
            )
            continue
        candidates = [
            s for (st, lay, s) in capture_b.maps if st == stream and lay == layer
        ]
        candidates = [s for s in candidates if s in time_b]
        if not candidates:
            rows.append(
                {
                    "stream": stream,
                    "layer": layer,
                    "step_a": step_a,
                    "timestep_a": round(t_a, 2),
                    "skipped": f"run B captured nothing for {stream} block {layer}",
                }
            )
            continue
        step_b = min(candidates, key=lambda j: (abs(time_b[j] - t_a), j))
        t_b = time_b[step_b]
        gap = abs(t_b - t_a)
        base = {
            "stream": stream,
            "layer": layer,
            "timestep_a": round(t_a, 2),
            "timestep_b": round(t_b, 2),
            "timestep_gap": round(gap, 2),
            "step_a": step_a,
            "step_b": step_b,
        }
        if gap > max_gap:
            rows.append(
                {
                    **base,
                    "skipped": (
                        f"nearest captured timestep in run B is {t_b:.2f}, a gap of "
                        f"{gap:.2f} beyond the {max_gap:.2f} tolerance"
                    ),
                }
            )
            continue
        a = capture_a.maps[(stream, layer, step_a)]
        b = capture_b.maps[(stream, layer, step_b)]
        try:
            contrasts = attention_contrast(a, b, min_effect=min_effect)
        except ValueError as exc:
            rows.append({**base, "skipped": str(exc)})
            continue
        ranked = head_ranking(contrasts)
        rows.append(
            {
                **base,
                "n_heads_changed": len(contrasts),
                "top_heads": [c.summary() for c in ranked[:5]],
                "max_effect": ranked[0].effect_size if ranked else 0.0,
                "effect_kind": ranked[0].effect_kind if ranked else None,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from omegaconf import OmegaConf
    from ovi.ovi_fusion_engine import OviFusionEngine

    from wsp.dit import CaptureSchedule, ovi_sampler_kwargs

    config = OmegaConf.load(args.config)
    if args.max_frames is not None:
        config.max_frames = args.max_frames

    sampler = ovi_sampler_kwargs(config, args.negative_prompt)
    print(f"sampler: {sampler}", flush=True)

    print("loading engine...", flush=True)
    engine = OviFusionEngine(config=config, device=0)
    print("engine loaded", flush=True)

    reports: dict[str, dict] = {}
    captures: dict[int, object] = {}
    failures: list[str] = []
    steps_used: dict[str, list[int] | None] = {}

    for steps in args.rollouts:
        print(f"\n=== rollout: {steps} steps, seed {args.seed} ===", flush=True)
        chosen = capture_steps_for(args, steps)
        steps_used[str(steps)] = None if chosen is None else list(chosen)
        schedule = CaptureSchedule(
            layers=tuple(args.layers),
            timesteps=chosen,
            keep_patches=False,
            streams=tuple(args.streams),
        )
        print(f"schedule: {schedule.describe()}", flush=True)
        try:
            report, capture, _video, _audio = run_rollout(
                engine, args, steps, schedule, sampler
            )
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
        "streams": list(args.streams),
        "capture_steps": steps_used,
        "cfg_pass": args.cfg_pass,
        "sampler": sampler,
        "chunk": args.chunk,
        "rollouts": reports,
        "summaries": {str(s): _summaries(c) for s, c in captures.items()},
    }

    # Contrast the first two rollouts, which is the drifting-versus-stable pair.
    if len(captures) >= 2:
        ordered = sorted(captures)
        a, b = ordered[0], ordered[1]
        rows = _contrast(
            captures[a],
            captures[b],
            reports[str(a)].get("timesteps", {}),
            reports[str(b)].get("timesteps", {}),
            min_effect=0.05,
            max_gap=args.max_gap,
        )
        payload["contrast"] = {
            "a_steps": a,
            "b_steps": b,
            "aligned_by": "timestep (captured steps only)",
            "min_effect": 0.05,
            "n_rows": len(rows),
            "n_compared": sum(1 for r in rows if "skipped" not in r),
            "n_skipped": sum(1 for r in rows if "skipped" in r),
            "rows": rows,
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
