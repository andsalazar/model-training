"""Derive WSP directions from a capture and project them onto Ovi's weight columns.

The step this project has been building toward: read two rollouts of the same prompt and
seed that differ only in step count, subtract their residual states to get a direction,
and ask which weight columns in the residual writers align with it.

Why the direction is the interesting object
-------------------------------------------
A direction is a *contrast*, and the pair is controlled: 20 steps drifted colour in 2 of
3 seeds, 30 steps was stable in 3 of 3. So "stable minus drifting" is a direction in
which the model would have to move to be stable. If a small set of weight columns can
move it there, that is the surgical edit WSP is supposed to find.

Aligning by noise level, not by step index
------------------------------------------
20 steps and 30 steps visit different timesteps. Step 3 of 20 is *not* the same point on
the trajectory as step 3 of 30. The capture records the timestep behind each step index,
so the pairing is done by nearest timestep value. Pairing by index would subtract two
unrelated states and the direction would be arithmetic noise that still has unit length
and still produces alignment scores.

Output
------
A JSON report with, per (stream, layer, paired step): the direction norm before
normalisation (magnitude is information -- a near-zero norm means the two runs agree),
the cosine between the two states, and the top aligned weight columns. Plus the
TargetPlan that the edit would use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wsp.activations import alignment_spectrum
from wsp.dit import _windowed_mean


def load_capture(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load the npz of arrays and its sibling JSON report."""
    arrays: dict[str, np.ndarray] = {}
    with np.load(path) as data:
        for key in data.files:
            arrays[key] = data[key]
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return arrays, meta


def parse_capture_key(key: str) -> tuple[int, str, int, int] | None:
    """``capture_<steps>_<stream>_<layer>_<step>`` -> its four parts."""
    if not key.startswith("capture_"):
        return None
    parts = key.split("_")
    if len(parts) != 5:
        return None
    try:
        return int(parts[1]), parts[2], int(parts[3]), int(parts[4])
    except ValueError:
        return None


def pair_by_timestep(
    meta: dict, steps_a: int, steps_b: int
) -> list[tuple[int, int, float, float]]:
    """Match step indices across two rollouts by nearest timestep value.

    Returns ``(step_a, step_b, timestep_a, timestep_b)`` triples. A capture that did not
    record its timesteps cannot be paired -- refusing is the point, because pairing by
    index would produce a plausible direction from two different points on the
    trajectory.
    """
    map_a = meta.get(str(steps_a), {}).get("step_timesteps")
    map_b = meta.get(str(steps_b), {}).get("step_timesteps")
    if not map_a or not map_b:
        raise ValueError(
            f"rollout {steps_a} or {steps_b} did not record step_timesteps; "
            "cannot align by noise level, and aligning by index would subtract "
            "states from different points on the trajectory"
        )
    time_a = {int(k): float(v) for k, v in map_a.items()}
    time_b = {int(k): float(v) for k, v in map_b.items()}
    pairs: list[tuple[int, int, float, float]] = []
    for ia, ta in sorted(time_a.items()):
        ib = min(time_b, key=lambda j: abs(time_b[j] - ta))
        tb = time_b[ib]
        pairs.append((ia, ib, ta, tb))
    return pairs


def derive(
    arrays: dict[str, np.ndarray],
    meta: dict,
    steps_a: int,
    steps_b: int,
    *,
    stream: str = "video",
    n_frames: int | None = None,
    frames: str | None = None,
    depth: int = 20,
) -> dict:
    """Contrast ``steps_b`` against ``steps_a`` at every captured layer.

    The direction points from ``steps_a`` toward ``steps_b``; callers should pass the
    drifting run first so the direction points toward stability.
    """
    pairs = pair_by_timestep(meta, steps_a, steps_b)
    layers = sorted(
        {
            layer
            for key in arrays
            for parsed in [parse_capture_key(key)]
            if parsed
            for run, s, layer, _ in [parsed]
            if run == steps_a and s == stream
        }
    )
    results: list[dict] = []
    for layer in layers:
        for ia, ib, ta, tb in pairs:
            key_a = f"capture_{steps_a}_{stream}_{layer}_{ia}"
            key_b = f"capture_{steps_b}_{stream}_{layer}_{ib}"
            if key_a not in arrays or key_b not in arrays:
                continue
            matrix_a = np.asarray(arrays[key_a], dtype=np.float32)
            matrix_b = np.asarray(arrays[key_b], dtype=np.float32)
            if matrix_a.shape != matrix_b.shape:
                results.append(
                    {
                        "layer": layer,
                        "step_a": ia,
                        "step_b": ib,
                        "skipped": (
                            f"shape mismatch {matrix_a.shape} vs {matrix_b.shape}; "
                            "the two rollouts captured different token counts"
                        ),
                    }
                )
                continue
            mean_a = _windowed_mean(matrix_a, frames, n_frames or 1, stream)
            mean_b = _windowed_mean(matrix_b, frames, n_frames or 1, stream)
            delta = mean_b - mean_a
            norm = float(np.linalg.norm(delta))
            denom = float(
                np.linalg.norm(mean_a) * np.linalg.norm(mean_b)
            )
            cosine = float(mean_a @ mean_b / denom) if denom > 0 else float("nan")
            entry: dict = {
                "layer": layer,
                "step_a": ia,
                "step_b": ib,
                "timestep_a": ta,
                "timestep_b": tb,
                "delta_norm": norm,
                "cosine": cosine,
                "state_norm_a": float(np.linalg.norm(mean_a)),
                "state_norm_b": float(np.linalg.norm(mean_b)),
            }
            if norm > 0:
                entry["direction_norm_after_unit"] = 1.0
                entry["direction"] = (delta / norm).astype(np.float32)
            else:
                entry["note"] = "states identical; no direction at this (layer, step)"
            results.append(entry)
    return {
        "stream": stream,
        "steps_a": steps_a,
        "steps_b": steps_b,
        "pairs": len(pairs),
        "layers": layers,
        "results": results,
    }


def summarize(direction_report: dict, depth: int = 20) -> dict:
    """A JSON-safe summary: drop the raw vectors, keep the measurable facts."""
    out: list[dict] = []
    for entry in direction_report["results"]:
        row = {k: v for k, v in entry.items() if k != "direction"}
        out.append(row)
    norms = [r["delta_norm"] for r in out if "delta_norm" in r]
    return {
        "stream": direction_report["stream"],
        "steps_a": direction_report["steps_a"],
        "steps_b": direction_report["steps_b"],
        "layers": direction_report["layers"],
        "n_pairs": direction_report["pairs"],
        "n_directions": sum(1 for r in out if "delta_norm" in r and r["delta_norm"] > 0),
        "n_identical": sum(1 for r in out if r.get("delta_norm") == 0.0),
        "n_skipped": sum(1 for r in out if "skipped" in r),
        "delta_norm_min": min(norms) if norms else None,
        "delta_norm_max": max(norms) if norms else None,
        "delta_norm_median": float(np.median(norms)) if norms else None,
        "rows": out,
    }


def project(
    direction_report: dict,
    weights: dict[tuple[int, str], np.ndarray],
    *,
    depth: int = 10,
) -> list[dict]:
    """Score ``weights[(layer, key)]`` columns against each direction."""
    findings: list[dict] = []
    for entry in direction_report["results"]:
        direction = entry.get("direction")
        if direction is None:
            continue
        layer = entry["layer"]
        for (w_layer, key), matrix in sorted(weights.items()):
            if w_layer != layer:
                continue
            matrix = np.asarray(matrix, dtype=np.float32)
            if matrix.shape[0] != direction.shape[0]:
                findings.append(
                    {
                        "layer": layer,
                        "step": entry["step_b"],
                        "key": key,
                        "skipped": (
                            f"direction has {direction.shape[0]} dims but weight has "
                            f"{matrix.shape[0]} rows"
                        ),
                    }
                )
                continue
            spectrum = alignment_spectrum(matrix, direction, depth=depth)
            scores = [s for _, s in spectrum]
            findings.append(
                {
                    "layer": layer,
                    "step_a": entry["step_a"],
                    "step_b": entry["step_b"],
                    "key": key,
                    "top_columns": [int(i) for i, _ in spectrum],
                    "top_scores": [round(float(s), 6) for s in scores],
                    "best": round(float(scores[0]), 6) if scores else None,
                    "spread_top_to_k": (
                        round(float(scores[0] - scores[-1]), 6) if scores else None
                    ),
                }
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps-a", type=int, required=True, help="drifting run")
    parser.add_argument("--steps-b", type=int, required=True, help="stable run")
    parser.add_argument("--stream", default="video", choices=["video", "audio"])
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument("--frames", default=None, help="frame window, e.g. 0:1")
    args = parser.parse_args(argv)

    arrays, meta = load_capture(args.capture)
    print(f"arrays: {len(arrays)}; rollouts in report: {sorted(meta)}", flush=True)

    report = derive(
        arrays,
        meta,
        args.steps_a,
        args.steps_b,
        stream=args.stream,
        n_frames=args.n_frames,
        frames=args.frames,
    )
    summary = summarize(report)
    print(json.dumps(summary, indent=2, default=str), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
