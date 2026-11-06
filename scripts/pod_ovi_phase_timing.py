"""Break a generation's wall-clock time into denoise vs decode.

Why this matters for the speed goal
-----------------------------------
Cutting denoising steps only buys time proportional to the denoise share. The
previous stage's 5-second run showed 30 steps at ~12s/step (about 6 minutes) followed
by a decode tail reported as 446s (~7.4 minutes). If that tail really is half the
wall clock, then halving the steps saves only ~25% of total time, not 50%.

That single number decides whether the step sweep is worth running at all, and it costs
one generation to measure. So this runs first.

Method
------
Parses Ovi's own tqdm output. The denoise loop prints `N/50 [MM:SS<MM:SS, Xs/it]`
style lines; the decode phase prints its own progress after. Rather than trust the
format, this records the wall clock at each phase transition by watching the log while
the process runs, and reports the split directly.

Usage on the pod
----------------
    source /workspace/ovi-env/bin/activate
    python pod_ovi_phase_timing.py --steps 50 --model 960x960_5s
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

#: Prompt reused from the speed sweep so the timing is directly comparable.
PROMPT = (
    "A river flowing through a forest, camera slowly panning. "
    "Audio: gentle water sounds and birdsong."
)

#: tqdm writes `\r`-delimited updates. A line matching this is a denoise step.
_STEP_RE = re.compile(r"(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)(s/it|it/s)\]")


def _write_config(
    path: Path, *, model: str, steps: int, output_dir: Path, seed: int = 100
) -> None:
    safe = PROMPT.replace('"', '\\"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""ckpt_dir: /workspace/Ovi/ckpts
output_dir: {output_dir}
sample_steps: {steps}
solver_name: unipc
model_name: "{model}"
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
video_frame_height_width: [960, 960]
text_prompt: "{safe}"
slg_layer: 11
each_example_n_times: 1
"""
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="960x960_5s")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--work", default="/workspace/ovi_phase")
    ap.add_argument("--out", default="/workspace/phase_timing.json")
    args = ap.parse_args()

    work = Path(args.work)
    out_dir = work / f"{args.model}_{args.steps}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = work / f"cfg_{args.model}_{args.steps}.yaml"
    _write_config(cfg, model=args.model, steps=args.steps, output_dir=out_dir)

    log_path = work / f"log_{args.model}_{args.steps}.txt"
    cmd = [sys.executable, "inference.py", "--config-file", str(cfg)]

    started = time.time()
    last_step_seen: int | None = None
    denoise_end: float | None = None
    per_step: list[float] = []

    with log_path.open("w") as log, subprocess.Popen(
        cmd, cwd="/workspace/Ovi", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            # tqdm uses \r, so one "line" can contain many updates.
            for match in _STEP_RE.finditer(line):
                current = int(match.group(1))
                rate = float(match.group(5))
                if match.group(6) == "s/it":
                    per_step.append(rate)
                if last_step_seen is None or current > last_step_seen:
                    last_step_seen = current
                if current == args.steps and denoise_end is None:
                    denoise_end = time.time()
        proc.wait()
    total = time.time() - started

    if denoise_end is None:
        # The final step line may not have been emitted; fall back to the sum of the
        # per-step rates, which is what tqdm would have converged to.
        estimated = sum(per_step[-args.steps :]) if per_step else None
        denoise_seconds = estimated
    else:
        denoise_seconds = denoise_end - started

    decode_seconds = (total - denoise_seconds) if denoise_seconds is not None else None
    report = {
        "model": args.model,
        "steps": args.steps,
        "total_seconds": round(total, 1),
        "denoise_seconds": round(denoise_seconds, 1) if denoise_seconds else None,
        "decode_seconds": round(decode_seconds, 1) if decode_seconds else None,
        "per_step_seconds": round(sum(per_step) / len(per_step), 2) if per_step else None,
        "steps_observed": last_step_seen,
    }
    if denoise_seconds and decode_seconds:
        report["denoise_share"] = round(denoise_seconds / total, 3)
        report["decode_share"] = round(decode_seconds / total, 3)
        # The headline number: what fraction of total time a 50% step cut would save.
        report["saving_from_halving_steps"] = round(
            (denoise_seconds * 0.5) / total, 3
        )

    print(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
