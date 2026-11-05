"""Measure the model's real behaviour on the task prompts, with a large cap.

The loop premise was wrong: at a 2048-token cap the model terminates. This
measures what it actually does -- token counts, whether it reaches the correct
answer, and how much of the budget goes to thinking -- so the next run targets a
real behaviour rather than a truncation artifact.
"""

from __future__ import annotations

import sys

MODEL = "/workspace/qwen"

PROMPTS = [
    "What is 17 times 24? Show your work step by step.",
    "What is 12 times 11?",
    "What is 2 plus 2?",
    "What is 391 divided by 17?",
    "What is 48 times 37?",
]


def main() -> int:
    from wsp.runner import VLLMRunner

    runner = VLLMRunner(MODEL)
    runner._ensure_loaded()

    for prompt in PROMPTS:
        signal = runner.measure(prompt, max_tokens=2048)
        text = runner.generate(prompt, max_tokens=2048)
        print(f"prompt={prompt!r}")
        print(f"  looped={signal['looped']} hit_cap={signal['hit_cap']} "
              f"repeated={signal['repeated']} tokens={signal['tokens']} "
              f"chars={signal['text_chars']}")
        print(f"  head: {text[:180]!r}")
        print(f"  tail: {text[-180:]!r}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
