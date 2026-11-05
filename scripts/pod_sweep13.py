"""Sweep the explore13 directional edit's scale and measure what it generalises to.

explore13's single edit cut the target prompt from 2048 to 191 tokens, but an
independent re-measurement showed a held-out question regressed from 830 to 2048.
That is the difference between "suppressed over-thinking" and "suppressed this
prompt". This script answers which one it is by sweeping the scale and measuring
several prompts at each setting.

It reuses the captured direction by re-capturing it deterministically (same
prompts, same layer, seed 0), so nothing depends on the editor's in-memory state.

Usage:
    python pod_sweep13.py <checkpoint> <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

import numpy as np

from wsp.activations import direction_from_contrast
from wsp.runner import TransformersActivationCapturer, VLLMRunner, classify
from wsp.weights.store import WeightStore

LAYER = 12
TENSOR = f"model.language_model.layers.{LAYER}.mlp.down_proj.weight"
LOOPING = "What is 12 times 11? Reply with only the digits."
CONTROL = "What is 12 times 11? Answer with just the number."

#: Columns explore13's aligned_columns ranked highest at layer 12.
COLS = "900:1000"

PROBES = [
    ("target", "What is 12 times 11? Reply with only the digits.", "132"),
    ("control", "What is 12 times 11? Answer with just the number.", "132"),
    ("heldout_17x23", "What is 17 times 23? Reply with only the digits.", "391"),
    ("heldout_13x14", "What is 13 times 14? Reply with only the digits.", "182"),
    ("heldout_9x8", "What is 9 times 8? Reply with only the digits.", "72"),
    ("nonmath", "Name the capital of France. Reply with only the city.", "Paris"),
]

SCALES = [0.0, 0.5, 1.0, 2.0]
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
            f"  {label:>5s} {name:14s} tokens={signal.tokens:5d} "
            f"hit_cap={str(signal.hit_cap):5s} correct={row['correct']}",
            flush=True,
        )
    return rows


def main() -> int:
    checkpoint = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/sweep13.json"

    print(f"checkpoint: {checkpoint}", flush=True)
    print(f"tensor: {TENSOR}  cols: {COLS}", flush=True)

    # Capture the direction once, deterministically.
    print("\ncapturing direction...", flush=True)
    capturer = TransformersActivationCapturer(checkpoint)
    looping = capturer.capture(LOOPING, TENSOR, max_tokens=64)
    control = capturer.capture(CONTROL, TENSOR, max_tokens=64)
    direction = direction_from_contrast(looping, control)
    print(f"  direction dim {direction.shape[0]}  norm {np.linalg.norm(direction):.4f}", flush=True)

    # WeightStore opens a safetensors file, not a directory.
    ckpt_file = next(Path(checkpoint).glob("*.safetensors"))
    store = WeightStore(ckpt_file)
    original = store.get(store.resolve(TENSOR)).copy()

    runner = VLLMRunner(checkpoint)
    all_rows: list[dict] = []

    for scale in SCALES:
        label = f"s{scale}"
        print(f"\n=== scale {scale} ===", flush=True)
        if scale == 0.0:
            # Baseline: the untouched checkpoint.
            all_rows.extend(measure_all(runner, label))
            continue

        edited = original.copy()
        lo, hi = (int(x) for x in COLS.split(":"))
        edited[:, lo:hi] -= scale * direction[:, None]

        ckpt_dir = Path(f"/workspace/sweep_L{LAYER}_{label}")
        from wsp.runner import write_edited_checkpoint

        ckpt_dir = write_edited_checkpoint(store.path, TENSOR, edited, ckpt_dir)
        changed = int(np.count_nonzero(~np.isclose(edited, original, rtol=0.0, atol=0.0)))
        print(f"  wrote {ckpt_dir} ({changed} weights changed)", flush=True)
        runner.reload(str(ckpt_dir))
        all_rows.extend(measure_all(runner, label))

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(all_rows, fh, indent=2)
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
