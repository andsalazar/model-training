"""Quantify the real behaviour: over-thinking, not looping.

The model reaches correct answers and terminates, but "What is 12 times 11?"
ran to a 2048-token cap without stopping. That is the behaviour worth targeting:
excessive self-verification on a trivial problem. This measures how much of the
budget goes to thinking and whether the answer appears before the cap.
"""

from __future__ import annotations

import sys

MODEL = "/workspace/qwen"

PROMPTS = [
    "What is 12 times 11?",
    "What is 12 times 11? Answer with just the number.",
    "What is 12 times 11? Reply with only the digits.",
]


def main() -> int:
    from wsp.runner import VLLMRunner

    runner = VLLMRunner(MODEL)
    runner._ensure_loaded()

    for prompt in PROMPTS:
        signal = runner.measure(prompt, max_tokens=2048)
        text = runner.generate(prompt, max_tokens=2048)
        # Does the correct answer appear at all?
        has_answer = "132" in text
        print(f"prompt={prompt!r}")
        print(f"  looped={signal['looped']} hit_cap={signal['hit_cap']} "
              f"tokens={signal['tokens']} chars={signal['text_chars']}")
        print(f"  answer 132 present: {has_answer}")
        print(f"  tail: {text[-200:]!r}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
