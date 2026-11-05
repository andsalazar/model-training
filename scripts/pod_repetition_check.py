"""Check whether the repetition detector fires on genuinely looping text.

Every "loop" reported so far had repeated=false: the model hit a small token cap
while thinking at length, which is truncation, not looping. If the detector never
fires on real looping text either, then no run has ever observed a loop and the
task premise needs revisiting. This tests the detector against text that really
does circle, and against long varied text that does not.
"""

from __future__ import annotations

import sys


def main() -> int:
    from wsp.runner import classify, detect_repetition

    cases = [
        (
            "verbatim block repeated",
            "Let me verify this calculation once more. The answer should be 408. " * 12,
        ),
        (
            "alternating pair",
            "Wait, let me reconsider. Actually, let me check. " * 20,
        ),
        (
            "long varied reasoning",
            " ".join(f"Step {i}: consider partial product {i * 7}." for i in range(120)),
        ),
        (
            "single sentence repeated",
            "I need to double-check that. " * 40,
        ),
    ]

    for label, text in cases:
        repeated = detect_repetition(text)
        signal = classify(text, n_tokens=len(text) // 4, max_tokens=4096)
        print(f"{label}:")
        print(f"  chars={len(text)} repeated={repeated} looped={signal.looped}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
