"""Independently measure explore14's two edits.

explore14's editor applied two directional edits and then reported post-edit
measurements without calling measure_loop after either one. This script measures
the real behaviour of both edited checkpoints so the claim can be settled.

Usage:
    python pod_verify14.py <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

from wsp.runner import VLLMRunner, classify

PROBES = [
    ("target", "What is 12 times 11? Reply with only the digits.", "132"),
    ("control", "What is 12 times 11? Answer with just the number.", "132"),
    ("heldout_17x23", "What is 17 times 23? Reply with only the digits.", "391"),
    ("heldout_7x8", "What is 7 times 8? Reply with only the digits.", "56"),
    ("nonmath", "Name the capital of France. Reply with only the city.", "Paris"),
]

MAX_TOKENS = 2048

CHECKPOINTS = [
    ("baseline", "/workspace/qwen"),
    ("edit_l10_s1", "/tmp/edit_l10_s1"),
    ("edit_l10_b_s1", "/tmp/edit_l10_b_s1"),
]


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/verify14.json"
    all_rows = []

    for label, path in CHECKPOINTS:
        if not Path(path).exists():
            print(f"SKIP {label}: {path} does not exist", flush=True)
            continue
        print(f"\n########## {label} ({path}) ##########", flush=True)
        runner = VLLMRunner(path)
        for name, prompt, expected in PROBES:
            text = runner.generate(prompt, max_tokens=MAX_TOKENS)
            n_tokens = len(runner._tokenizer(text).input_ids)
            signal = classify(text, n_tokens, MAX_TOKENS)
            row = {
                "checkpoint": label,
                "probe": name,
                "prompt": prompt,
                "expected": expected,
                "correct": expected.lower() in text.lower(),
                **signal.as_dict(),
            }
            all_rows.append(row)
            print(
                f"  {name:14s} tokens={signal.tokens:5d} "
                f"hit_cap={str(signal.hit_cap):5s} correct={row['correct']}",
                flush=True,
            )

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(all_rows, fh, indent=2)
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
