"""Edit only the columns the alignment ranking identified, and measure spillover.

explore13's edit used the contiguous window 900:1000 because that was the only
form the tool accepted. Only 3 of the 12 columns its own ranking identified fell
inside that window, so 97 edited columns were never ranked as aligned. The scale
sweep showed the edit redistributes over-thinking rather than suppressing it.

This script tests the obvious alternative: edit exactly the ranked columns. If
the target improves without the held-out spillover, the window was the problem.
If the spillover persists, a single-layer directional edit is the wrong
abstraction.

Usage:
    python pod_sweep_ranked.py <checkpoint> <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

import numpy as np

from wsp.activations import direction_from_contrast, top_aligned_columns
from wsp.runner import (
    TransformersActivationCapturer,
    VLLMRunner,
    classify,
    write_edited_checkpoint,
)
from wsp.weights.store import WeightStore

LAYER = 12
TENSOR = f"model.language_model.layers.{LAYER}.mlp.down_proj.weight"
LOOPING = "What is 12 times 11? Reply with only the digits."
CONTROL = "What is 12 times 11? Answer with just the number."

PROBES = [
    ("target", "What is 12 times 11? Reply with only the digits.", "132"),
    ("control", "What is 12 times 11? Answer with just the number.", "132"),
    ("heldout_17x23", "What is 17 times 23? Reply with only the digits.", "391"),
    ("heldout_13x14", "What is 13 times 14? Reply with only the digits.", "182"),
    ("heldout_9x8", "What is 9 times 8? Reply with only the digits.", "72"),
    ("nonmath", "Name the capital of France. Reply with only the city.", "Paris"),
]

#: How many top-ranked columns to edit. 12 is what explore13's ranking returned.
TOP_K = 12
SCALES = [0.5, 1.0]
MAX_TOKENS = 2048


def measure_all(runner: VLLMRunner, label: str) -> list[dict]:
    rows = []
    for name, prompt, expected in PROBES:
        text = runner.generate(prompt, max_tokens=MAX_TOKENS)
        n_tokens = len(runner._tokenizer(text).input_ids)
        signal = classify(text, n_tokens, MAX_TOKENS)
        row = {
            "scale": label,
            "probe": name,
            "prompt": prompt,
            "expected": expected,
            "correct": expected.lower() in text.lower(),
            **signal.as_dict(),
        }
        rows.append(row)
        print(
            f"  {label:>10s} {name:14s} tokens={signal.tokens:5d} "
            f"hit_cap={str(signal.hit_cap):5s} correct={row['correct']}",
            flush=True,
        )
    return rows


def main() -> int:
    checkpoint = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/sweep_ranked.json"

    print(f"checkpoint: {checkpoint}", flush=True)
    print(f"tensor: {TENSOR}  top_k: {TOP_K}", flush=True)

    print("\ncapturing direction...", flush=True)
    capturer = TransformersActivationCapturer(checkpoint)
    looping = capturer.capture(LOOPING, TENSOR, max_tokens=64)
    control = capturer.capture(CONTROL, TENSOR, max_tokens=64)
    direction = direction_from_contrast(looping, control)
    print(f"  direction dim {direction.shape[0]}", flush=True)

    ckpt_file = next(Path(checkpoint).glob("*.safetensors"))
    store = WeightStore(ckpt_file)
    original = store.get(store.resolve(TENSOR)).copy()

    ranked = top_aligned_columns(original, direction, k=TOP_K)
    cols = [int(c) for c, _ in ranked]
    print(f"  ranked columns: {cols}", flush=True)
    print(f"  cosines: {[round(float(s), 4) for _, s in ranked]}", flush=True)

    runner = VLLMRunner(checkpoint)
    all_rows: list[dict] = []

    for scale in SCALES:
        label = f"ranked_s{scale}"
        print(f"\n=== {label} ===", flush=True)
        edited = original.copy()
        edited[:, cols] -= scale * direction[:, None]

        ckpt_dir = Path(f"/workspace/ranked_L{LAYER}_{scale}")
        ckpt_dir = write_edited_checkpoint(store.path, TENSOR, edited, ckpt_dir)
        changed = int(np.count_nonzero(~np.isclose(edited, original, rtol=0.0, atol=0.0)))
        print(f"  wrote {ckpt_dir} ({changed} weights changed)", flush=True)
        runner.reload(str(ckpt_dir))
        all_rows.extend(measure_all(runner, label))

    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"cols": cols, "rows": all_rows}, fh, indent=2)
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
