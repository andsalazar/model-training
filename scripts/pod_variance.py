"""How much does one measurement of a prompt vary?

Stage 2's probe-set test measured the same checkpoint twice and got wildly
different baselines from Stage 1: 13x14 was 827 tokens in Stage 1 and 259 here;
9x8 was 1309 and 181. Those are 3-7x differences on an UNMODIFIED model.

The verifier compares one baseline reading against one edited reading. If a
single reading varies by 3x, then "improved" and "regressed" can both be drawn
from noise, and every verdict in Stage 1 and explore15 is suspect.

This measures the same prompt N times on the same checkpoint and reports the
spread, so the noise floor can be established empirically instead of guessed at.

Run as a file, not a heredoc: vLLM spawns a core process that re-imports main.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

from wsp.runner import VLLMRunner, classify

MODEL = "/local/models/qwen"
MAX_TOKENS = 2048
REPEATS = 7

PROMPTS = [
    ("target_12x11", "What is 12 times 11? Reply with only the digits.", "132"),
    ("13x14", "What is 13 times 14? Reply with only the digits.", "182"),
    ("9x8", "What is 9 times 8? Reply with only the digits.", "72"),
    ("17x23", "What is 17 times 23? Reply with only the digits.", "391"),
    ("guard_capital", "Name the capital of France. Reply with only the city.", "Paris"),
]


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/variance.json"
    runner = VLLMRunner(MODEL)
    results = {}

    for name, prompt, expected in PROMPTS:
        tokens = []
        caps = 0
        wrong = 0
        for i in range(REPEATS):
            text = runner.generate(prompt, max_tokens=MAX_TOKENS)
            n = runner.count_tokens(text)
            signal = classify(text, n, MAX_TOKENS)
            tokens.append(signal.tokens)
            caps += int(signal.hit_cap)
            wrong += int(expected.lower() not in text.lower())
            print(f"  {name:14s} run {i + 1}/{REPEATS}: {signal.tokens:5d} tokens", flush=True)

        lo, hi = min(tokens), max(tokens)
        results[name] = {
            "tokens": tokens,
            "min": lo,
            "max": hi,
            "mean": statistics.mean(tokens),
            "median": statistics.median(tokens),
            "stdev": statistics.stdev(tokens) if len(tokens) > 1 else 0.0,
            "ratio_max_min": (hi / lo) if lo else None,
            "cap_hits": caps,
            "wrong": wrong,
            "expected": expected,
        }
        print(
            f"{name:14s} min={lo:5d} max={hi:5d} mean={results[name]['mean']:7.1f} "
            f"stdev={results[name]['stdev']:6.1f} spread={results[name]['ratio_max_min']:.2f}x "
            f"caps={caps} wrong={wrong}",
            flush=True,
        )

    Path(out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
