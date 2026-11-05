"""Check whether short math prompts really loop, or just hit the token cap.

explore10's editor concluded that every math prompt loops, including "2 plus 2",
and therefore that no non-looping math control exists. That conclusion rests on
measure_loop reporting looped=true, but classify() defines looped as
``hit_cap or repeated``. A short prompt that answers correctly and then runs to a
small cap is reported as looping when it did not. This prints the actual text so
the distinction is visible.
"""

from __future__ import annotations

import sys

MODEL = "/workspace/qwen"

PROMPTS = [
    ("2 plus 2", 32),
    ("2 plus 2", 128),
    ("What is 12 times 11?", 64),
    ("What is 12 times 11?", 256),
    ("What is 17 times 24? Show your work step by step.", 256),
    ("Hello, how are you today?", 64),
]


def main() -> int:
    from wsp.runner import VLLMRunner

    runner = VLLMRunner(MODEL)
    runner._ensure_loaded()

    for prompt, cap in PROMPTS:
        signal = runner.measure(prompt, max_tokens=cap)
        text = runner.generate(prompt, max_tokens=cap)
        print(f"prompt={prompt!r} cap={cap}")
        print(f"  looped={signal['looped']} hit_cap={signal['hit_cap']} "
              f"repeated={signal['repeated']} tokens={signal['tokens']}")
        print(f"  text ({len(text)} chars): {text[:300]!r}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
