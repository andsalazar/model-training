"""Independently verify explore13's edit, without trusting the editor's prose.

Loads a checkpoint in vLLM and measures the same prompts the editor claimed to
measure, keeping the generated text so correctness can be checked as well as
termination. Run once per checkpoint and compare the two JSON files.

Usage:
    python pod_verify_edit13.py <checkpoint> <out.json>
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/workspace/src")

from wsp.runner import VLLMRunner, classify

PROMPTS = [
    ("target", "What is 12 times 11? Reply with only the digits.", 2048, "132"),
    ("control", "What is 12 times 11? Answer with just the number.", 2048, "132"),
    ("heldout", "What is 17 times 23? Reply with only the digits.", 2048, "391"),
    ("heldout2", "What is 13 times 14? Reply with only the digits.", 2048, "182"),
]


def main() -> int:
    checkpoint = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/verify13.json"

    print(f"checkpoint: {checkpoint}", flush=True)
    runner = VLLMRunner(checkpoint)
    results = []
    for label, prompt, cap, expected in PROMPTS:
        text = runner.generate(prompt, max_tokens=cap)
        # generate() returns text only, so count tokens by re-tokenising.
        n_tokens = len(runner._tokenizer(text).input_ids)
        signal = classify(text, n_tokens, cap)
        row = {
            "label": label,
            "prompt": prompt,
            "expected": expected,
            "correct": expected in text,
            **signal.as_dict(),
            "text": text,
        }
        results.append(row)
        print(
            f"{label:9s} tokens={signal.tokens:5d} hit_cap={signal.hit_cap} "
            f"repeated={signal.repeated} correct={row['correct']} "
            f"text={text[:100]!r}",
            flush=True,
        )

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
