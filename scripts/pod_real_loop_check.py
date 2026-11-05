"""Check whether the model genuinely loops, with a cap large enough to tell.

The previous check used small caps (32-256 tokens), where a verbose but
terminating answer also hits the cap. This uses a large cap so a real loop has
room to repeat, and prints the tail of the text so repetition is visible.
"""

from __future__ import annotations

import sys

MODEL = "/workspace/qwen"

PROMPTS = [
    "What is 17 times 24? Show your work step by step.",
    "What is 12 times 11?",
    "What is 2 plus 2?",
    "Hello, how are you today?",
]


def main() -> int:
    from wsp.runner import VLLMRunner, detect_repetition

    runner = VLLMRunner(MODEL)
    runner._ensure_loaded()

    for prompt in PROMPTS:
        signal = runner.measure(prompt, max_tokens=2048)
        text = runner.generate(prompt, max_tokens=2048)
        print(f"prompt={prompt!r}")
        print(f"  looped={signal['looped']} hit_cap={signal['hit_cap']} "
              f"repeated={signal['repeated']} tokens={signal['tokens']} "
              f"chars={signal['text_chars']}")
        print(f"  tail: {text[-260:]!r}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
