"""Measure the speed/quality trade-off across Ovi's denoising-step budget.

The question
------------
Ovi's default is 50 denoising steps. Fewer steps is the cheapest speed win available
and needs no model edit at all. The question is whether the *longest* sequence (10s)
tolerates a step reduction as well as the short one (5s) does — and if not, where the
two diverge.

Why steps and not something else
--------------------------------
The tunable surface in Ovi's config is: ``sample_steps``, ``solver_name``, ``shift``,
``video_guidance_scale``, ``audio_guidance_scale``, ``slg_layer``, ``fp8``,
``cpu_offload``. Of those, only ``sample_steps`` changes compute linearly. Guidance
scales change quality but not cost. ``fp8`` changes memory, not step count. So steps
is the only knob that directly buys wall-clock time.

What is measured
----------------
For each (horizon, steps) pair: wall-clock seconds, peak VRAM, and the four
reference-free quality metrics from ``wsp.quality``. Quality is judged as a ratio
against the same horizon at the default 50 steps, so the comparison is always
like-for-like.

The design deliberately measures 5s and 10s separately rather than assuming the
longer one behaves like the shorter one. That assumption is exactly what the earlier
stability work showed to be unsafe.

Usage on the pod
----------------
    source /workspace/ovi-env/bin/activate
    python pod_ovi_speed.py --out /workspace/speed.json

The script shells out to Ovi's own ``inference.py`` rather than importing its engine,
because the engine's API is not documented and the CLI is what the repo supports.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

#: The step budgets to test. 50 is Ovi's default and the quality reference. The rest
#: bracket it downward; 12 is aggressive enough that a failure is informative.
DEFAULT_STEPS = [50, 40, 30, 20, 12]

#: Horizons to test. Both are measured because the whole question is whether the long
#: one behaves differently from the short one.
DEFAULT_HORIZONS = ["960x960_5s", "960x960_10s"]

#: One prompt, reused across every condition, so quality differences are attributable
#: to steps and not to prompt content. Chosen for having both motion and detail, so a
#: quality loss shows up in more than one metric.
DEFAULT_PROMPT = (
    "A river flowing through a forest, camera slowly panning. "
    "Audio: gentle water sounds and birdsong."
)


def _read_frames(video_path: Path, max_frames: int | None = None) -> np.ndarray:
    """Decode an mp4 into ``(n_frames, h, w, 3)`` uint8."""
    try:
        import imageio.v3 as iio
    except ImportError:  # pragma: no cover - pod-only dependency
        import imageio as iio  # type: ignore[no-redef]

    frames = iio.imread(str(video_path), plugin="pyav")
    arr = np.asarray(frames)
    if arr.ndim == 4 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if max_frames is not None and len(arr) > max_frames:
        idx = np.linspace(0, len(arr) - 1, max_frames).astype(int)
        arr = arr[idx]
    return arr


def _write_config(
    path: Path,
    *,
    model_name: str,
    steps: int,
    prompt: str,
    output_dir: Path,
    seed: int,
    height_width: tuple[int, int] = (960, 960),
) -> None:
    """Write an Ovi inference YAML for one condition."""
    # The prompt contains a colon and commas, so it must be quoted for YAML.
    safe_prompt = prompt.replace('"', '\\"')
    text = f"""ckpt_dir: /workspace/Ovi/ckpts
output_dir: {output_dir}
sample_steps: {steps}
solver_name: unipc
model_name: "{model_name}"
shift: 5.0
sp_size: 1
audio_guidance_scale: 3.0
video_guidance_scale: 4.0
mode: "t2v"
fp8: False
cpu_offload: False
seed: {seed}
video_negative_prompt: "jitter, bad hands, blur, distortion"
audio_negative_prompt: "robotic, muffled, echo, distorted"
video_frame_height_width: [{height_width[0]}, {height_width[1]}]
text_prompt: "{safe_prompt}"
slg_layer: 11
each_example_n_times: 1
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _run_generation(config_path: Path, log_path: Path, timeout: int) -> dict:
    """Run Ovi's inference.py and return timing plus peak VRAM."""
    cmd = [
        sys.executable,
        "inference.py",
        "--config-file",
        str(config_path),
    ]
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd="/workspace/Ovi",
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    elapsed = time.time() - started

    text = log_path.read_text(errors="replace")
    peak = _parse_peak_vram(text)
    return {
        "returncode": proc.returncode,
        "seconds": round(elapsed, 2),
        "peak_vram_gb": peak,
        "log": str(log_path),
    }


def _parse_peak_vram(text: str) -> float | None:
    """Pull the largest 'X.XX GiB' or 'X MiB' figure out of a log, if present."""
    best = None
    for match in re.finditer(r"([\d.]+)\s*(GiB|MiB)", text):
        value = float(match.group(1))
        if match.group(2) == "MiB":
            value /= 1024.0
        if best is None or value > best:
            best = value
    return round(best, 2) if best is not None else None


def _find_video(output_dir: Path) -> Path | None:
    """Return the newest mp4 in a directory, or None."""
    videos = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    return videos[-1] if videos else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/workspace/speed.json")
    parser.add_argument("--work", default="/workspace/ovi_speed")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=DEFAULT_STEPS,
        help="step budgets to test",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=DEFAULT_HORIZONS,
        help="model variants to test",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=48,
        help="frames sampled per video for the quality metrics",
    )
    args = parser.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    # Imported here so the module can be read without the wsp package on the pod path.
    sys.path.insert(0, "/workspace/src")
    from wsp.quality import compare_quality, measure_quality  # noqa: PLC0415

    results: list[dict] = []
    for horizon in args.horizons:
        for steps in args.steps:
            tag = f"{horizon}_{steps}steps"
            out_dir = work / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg = work / f"cfg_{tag}.yaml"
            _write_config(
                cfg,
                model_name=horizon,
                steps=steps,
                prompt=args.prompt,
                output_dir=out_dir,
                seed=args.seed,
            )
            print(f"=== {tag} ===", flush=True)
            run = _run_generation(cfg, work / f"gen_{tag}.log", args.timeout)
            print(f"  {run['seconds']}s rc={run['returncode']}", flush=True)

            entry: dict = {
                "horizon": horizon,
                "steps": steps,
                "run": run,
            }
            video = _find_video(out_dir)
            if video is None:
                entry["error"] = "no mp4 produced"
                print("  NO VIDEO", flush=True)
                results.append(entry)
                continue

            frames = _read_frames(video, args.max_frames)
            reading = measure_quality(frames)
            entry["video"] = str(video)
            entry["n_frames"] = int(frames.shape[0])
            entry["quality"] = reading.as_dict()
            print(f"  quality {reading.as_dict()}", flush=True)
            results.append(entry)

    # Compare every condition against the same horizon at the default step count.
    default_steps = max(args.steps)
    by_key = {(r["horizon"], r["steps"]): r for r in results}
    comparisons: list[dict] = []
    for r in results:
        if "quality" not in r:
            continue
        base_entry = by_key.get((r["horizon"], default_steps))
        if base_entry is None or "quality" not in base_entry:
            continue
        from wsp.quality import QualityReading  # noqa: PLC0415

        base = QualityReading(**base_entry["quality"])
        cand = QualityReading(**r["quality"])
        cmp = compare_quality(base, cand)
        row = {
            "horizon": r["horizon"],
            "steps": r["steps"],
            "seconds": r["run"]["seconds"],
            "speedup_vs_default": round(
                base_entry["run"]["seconds"] / max(r["run"]["seconds"], 1e-9), 3
            ),
            "acceptable": cmp.acceptable,
            "ratios": {k: round(v, 4) for k, v in cmp.ratios.items()},
            "outside": cmp.outside,
            "reason": cmp.reason,
        }
        comparisons.append(row)
        print(f"  -> {row['speedup_vs_default']}x  acceptable={cmp.acceptable}", flush=True)

    report = {
        "prompt": args.prompt,
        "default_steps": default_steps,
        "results": results,
        "comparisons": comparisons,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
