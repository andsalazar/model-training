"""Check whether the looping/calm contrast actually separates the two states.

A contrast cosine near 1.0 means the two prompts produced nearly the same
activation state, so the derived direction is mostly noise. This measures the
separation directly and tries prompts that differ more sharply, so the editor
gets a direction that means something.
"""

from __future__ import annotations

import sys

MODEL = "/workspace/qwen"
TENSOR = "model.language_model.layers.23.mlp.down_proj.weight"


def main() -> int:
    import numpy as np

    from wsp.activations import contrast_cosine, direction_from_contrast
    from wsp.runner import TransformersActivationCapturer

    cap = TransformersActivationCapturer(MODEL, device="cuda", dtype="bfloat16")
    cap.reload(MODEL)
    print("model loaded\n")

    pairs = [
        (
            "short-loop",
            "Let me check. Let me check. Let me check. Let me check again.",
            "The answer is 42. Done.",
        ),
        (
            "long-loop",
            "Let me verify this. Let me verify this. Let me verify this. "
            "Let me verify this. Let me verify this. Let me verify this. "
            "Let me verify this. Let me verify this.",
            "Compute 17 * 23 = 391. The result is 391. Finished.",
        ),
        (
            "reasoning-vs-answer",
            "Wait, let me reconsider. Actually, let me reconsider. "
            "Hmm, let me reconsider. But wait, let me reconsider. "
            "Actually I should check again. Let me reconsider once more.",
            "391",
        ),
    ]

    for label, looping, calm in pairs:
        a = cap.capture(looping, TENSOR)
        b = cap.capture(calm, TENSOR)
        cos = contrast_cosine(a, b)
        vec = direction_from_contrast(a, b)
        # How much of the state's own magnitude does the direction represent?
        last = a[-1]
        rel = float(np.linalg.norm(vec)) / max(float(np.linalg.norm(last)), 1e-9)
        print(f"{label}:")
        print(f"  tokens {a.shape[0]} vs {b.shape[0]}")
        print(f"  contrast_cosine {cos:+.4f}")
        print(f"  direction norm {float(np.linalg.norm(vec)):.4f}")
        print(f"  relative to last-token norm {rel:.4f}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
