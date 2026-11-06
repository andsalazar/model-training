"""Parse an Ovi log into denoise vs decode time, without re-running anything.

A generation costs minutes of GPU, so being able to read the split out of an existing
log is worth more than a script that must regenerate to measure. This takes the log
`pod_ovi_phase_timing.py` writes and reports the breakdown.

Ovi's progress output (tqdm, `\\r`-delimited) looks like:

    50it [06:42,  8.05s/it]     <- denoise, one line per step, final one is total
    1it [07:15, 435.31s/it]     <- VAE decode, one line, huge per-iteration time

The decode phase emits exactly one `1it` line with a per-iteration time in the
hundreds of seconds, which is unmistakable against 8s/step denoising.

The `elapsed` field is CUMULATIVE from the start of the generate call, not
per-phase. That is the one thing easy to get wrong here: 402s of denoise plus a
decode line reading 435s does not mean 435s of decoding — it means 33s of decoding
after 402s of denoising. Reading it as per-phase gives a total larger than the
measured wall clock, which is how the error was caught. ``pod_ovi_verify_phases.sh``
confirms the interpretation empirically by comparing a 12-step and 50-step run.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: ``NNit [MM:SS, X.XXs/it]``
_ENTRY_RE = re.compile(r"(\d+)it\s*\[([\d:]+),\s*([\d.]+)(s/it|it/s)\]")

#: A per-iteration time above this is not a denoising step. Denoising runs at single
#: digit seconds per step on this hardware; decode is two orders of magnitude more.
_DECODE_THRESHOLD = 60.0


def _to_seconds(mmss: str) -> float:
    """``MM:SS`` or ``HH:MM:SS`` to seconds."""
    try:
        parts = [int(p) for p in mmss.split(":")]
    except ValueError as exc:
        raise ValueError(f"unrecognised time {mmss!r}") from exc
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"unrecognised time {mmss!r}")


def parse(path: Path) -> dict:
    text = path.read_text(errors="replace").replace("\r", "\n")
    entries = []
    for match in _ENTRY_RE.finditer(text):
        entries.append(
            {
                "count": int(match.group(1)),
                "elapsed": _to_seconds(match.group(2)),
                "rate": float(match.group(3)),
                "unit": match.group(4),
            }
        )

    if not entries:
        return {"error": "no progress entries found"}

    denoise = [e for e in entries if e["rate"] <= _DECODE_THRESHOLD and e["unit"] == "s/it"]
    decode = [e for e in entries if e["rate"] > _DECODE_THRESHOLD]

    # The last denoise entry's `elapsed` is the cumulative denoise time.
    denoise_seconds = max((e["elapsed"] for e in denoise), default=None)
    # The number of denoising steps is the largest counter seen, NOT the number of log
    # entries: tqdm re-renders its final line, so a 50-step run can produce 51 entries.
    # Using the entry count as the divisor made a step-count projection wrong by 10x.
    n_steps = max((e["count"] for e in denoise), default=None)
    # Decode's `elapsed` is ALSO cumulative from the start of generate, so the decode
    # duration is the difference. Treating it as per-phase would report a total far
    # larger than the wall clock.
    decode_marker = max((e["elapsed"] for e in decode), default=None)
    decode_seconds = (
        decode_marker - denoise_seconds
        if decode_marker is not None and denoise_seconds is not None
        else None
    )

    total = None
    if denoise_seconds is not None and decode_seconds is not None:
        total = denoise_seconds + decode_seconds

    report: dict = {
        "log": str(path),
        "denoise_entries": len(denoise),
        "decode_entries": len(decode),
        "steps": n_steps,
        "denoise_seconds": denoise_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": total,
        "per_step_seconds": (
            round(denoise_seconds / n_steps, 3)
            if n_steps and denoise_seconds
            else None
        ),
    }
    if total:
        report["denoise_share"] = round(denoise_seconds / total, 3)
        report["decode_share"] = round(decode_seconds / total, 3)
        # What a 50% step cut would actually save, as a share of total wall clock.
        report["saving_from_halving_steps"] = round((denoise_seconds * 0.5) / total, 3)
        # Steps needed to halve TOTAL time. Decode is fixed, so the target is
        #     new_denoise = total/2 - decode
        # and the step count scales linearly from the measured denoise time.
        #
        # An earlier version used `denoise - decode` here. That is wrong: for
        # denoise=3000s, decode=300s it gives 2700s of denoise (45 steps), but the
        # actual requirement is total/2 - decode = 1350s (22.5 steps). It understated
        # the cut needed by a factor of two and would have made a reachable target look
        # almost twice as hard as it is.
        target_denoise = total / 2.0 - decode_seconds
        if target_denoise <= 0:
            report["steps_for_2x_total"] = (
                "impossible - decode alone exceeds half of the total wall clock"
            )
        elif n_steps:
            report["steps_for_2x_total"] = round(
                n_steps * target_denoise / denoise_seconds, 1
            )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+")
    args = ap.parse_args()
    for log in args.logs:
        report = parse(Path(log))
        print(f"\n=== {Path(log).name} ===")
        for key, value in report.items():
            if key != "log":
                print(f"  {key}: {value}")


if __name__ == "__main__":
    sys.exit(main())
